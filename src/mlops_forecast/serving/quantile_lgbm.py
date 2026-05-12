"""Quantile-LightGBM pyfunc wrapper used at inference time.

Lives outside the Kedro pipelines package on purpose: when MLflow deserialises
the registered model it imports this module's class, and if the class lived
inside a Kedro pipeline package the import would drag Kedro into every API
container. Keeping serving code in its own minimal module is what lets us
ship a lean inference image (no Kedro, SHAP, GE, or Evidently in the request
path).

The wrapper bundles three LightGBM models:

* ``point_model``: standard regression head for the central forecast
* ``lower_model``: quantile-regression head at alpha=0.10
* ``upper_model``: quantile-regression head at alpha=0.90

Quantile regression alone tends to undercover (well-known LightGBM behaviour
on heteroscedastic data). To fix that we apply split conformal prediction
(Romano et al. 2019, "Conformalized Quantile Regression"): we hold out a
calibration set, compute how badly the raw quantile bounds miss it, and
widen the bounds at predict time by a fixed offset ``conformal_q``. The
resulting intervals have a finite-sample guarantee of marginal coverage at
the requested level, assuming the test data is exchangeable with the
calibration data.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


class QuantileLGBM(PythonModel):
    """Point + conformalised-quantile bundle, served as one model."""

    def __init__(
        self,
        point_model: lgb.LGBMRegressor,
        lower_model: lgb.LGBMRegressor,
        upper_model: lgb.LGBMRegressor,
        feature_names: list[str],
        conformal_q: float = 0.0,
    ) -> None:
        self.point_model = point_model
        self.lower_model = lower_model
        self.upper_model = upper_model
        self.feature_names = feature_names
        # Conformal offset: lower bounds are shifted down by this amount and
        # upper bounds shifted up. A value of 0 disables calibration.
        self.conformal_q = float(conformal_q)

    def predict(  # noqa: ARG002
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        # Reorder to match training column order; protects against caller bugs.
        X = model_input[self.feature_names]
        point = self.point_model.predict(X)
        raw_lower = self.lower_model.predict(X)
        raw_upper = self.upper_model.predict(X)
        return pd.DataFrame(
            {
                "load_mw_predicted": point,
                "lower_bound": raw_lower - self.conformal_q,
                "upper_bound": raw_upper + self.conformal_q,
            }
        )


def conformal_offset(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float,
) -> float:
    """Compute the split-conformal offset for a CQR interval.

    The non-conformity score for sample i is
        s_i = max(y_lower_i - y_i, y_i - y_upper_i)
    which is positive when the raw interval misses y_i and negative when it
    overshoots. The conformal offset is the (1 - alpha) quantile of the s_i,
    using the finite-sample correction from Romano et al. (2019):

        rank = ceil((n + 1)(1 - alpha)) / n

    Args:
        y_true: calibration targets, shape (n,).
        y_lower: raw lower-quantile predictions on the calibration set.
        y_upper: raw upper-quantile predictions on the calibration set.
        alpha: miscoverage rate (e.g. 0.2 for nominal 80% coverage).

    Returns:
        The amount to subtract from lower bounds and add to upper bounds.
        Non-negative for under-covered raw intervals; can go negative if the
        raw quantile heads already overcover.
    """
    n = len(y_true)
    scores = np.maximum(y_lower - y_true, y_true - y_upper)
    rank = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, rank, method="higher"))


def adaptive_conformal_offsets(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    target_coverage: float,
    gamma: float = 0.005,
    initial_q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream Adaptive Conformal Inference (ACI) over a chronological sequence.

    Implements the recursion from Gibbs and Candès (2021):

        alpha_{t+1} = alpha_t + gamma * (target_miscoverage - err_t)

    where err_t is 1 if y_t falls outside the predicted interval at time t and
    0 otherwise. At each step we recompute the conformal offset Q_t as the
    rank-(1 - alpha_t) empirical quantile of the non-conformity scores seen
    so far, then use Q_t to predict at time t+1.

    Unlike split-conformal calibration, ACI does not assume exchangeability
    between calibration and test data: it adjusts on the fly when coverage
    deviates from the target. This is the right tool when the test
    distribution shifts during inference, e.g. the COVID lockdown in the
    OPSD dataset.

    Args:
        y_true: chronologically ordered targets, shape (n,).
        y_lower: raw lower-quantile predictions, shape (n,).
        y_upper: raw upper-quantile predictions, shape (n,).
        target_coverage: nominal coverage level (e.g. 0.8 for 80%).
        gamma: learning rate. Smaller = slower adjustment, more stable.
        initial_q: starting offset, typically the split-conformal Q from
            training-time calibration.

    Returns:
        A tuple ``(q_history, hits)`` of length-n arrays. ``q_history[t]`` is
        the offset used to predict at time t; ``hits[t]`` is 1 if y_t was
        covered by the resulting interval and 0 otherwise.
    """
    n = len(y_true)
    target_mis = 1.0 - target_coverage
    alpha_t = target_mis
    q_history = np.zeros(n)
    hits = np.zeros(n, dtype=int)
    seen_scores: list[float] = []

    q_t = float(initial_q)
    for t in range(n):
        # Predict at time t with the current offset.
        lo = y_lower[t] - q_t
        hi = y_upper[t] + q_t
        covered = (y_true[t] >= lo) and (y_true[t] <= hi)
        q_history[t] = q_t
        hits[t] = int(covered)

        # Observe the score at time t and update the running miscoverage rate.
        s_t = max(y_lower[t] - y_true[t], y_true[t] - y_upper[t])
        seen_scores.append(float(s_t))
        err_t = 0.0 if covered else 1.0
        alpha_t = alpha_t + gamma * (target_mis - err_t)
        # alpha_t can drift outside [0, 1] in principle; clip so the quantile
        # call stays well-defined.
        alpha_t = float(np.clip(alpha_t, 1e-4, 1.0 - 1e-4))

        # Recompute Q from the running score history at the new alpha_t.
        if len(seen_scores) >= 2:
            q_t = float(np.quantile(seen_scores, 1.0 - alpha_t, method="higher"))

    return q_history, hits
