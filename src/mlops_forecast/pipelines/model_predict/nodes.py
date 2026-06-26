"""Score the held-out test set using whatever is in MLflow Production.

The pipeline never references a model artefact path on disk. It always
asks the registry for `models:/<name>/Production`, so promoting or rolling
back a model is a registry operation and not a code or container change.

Alongside the deployed point + calibrated-interval forecasts, we also
compute an Adaptive Conformal Inference (ACI) baseline as a what-if. The
deployed model uses a fixed split-conformal offset learned at training
time; ACI updates that offset on the fly as it observes coverage errors.
The comparison shows whether the static offset would still be appropriate
under the test-period distribution shift, or whether an adaptive scheme
would have been worth deploying.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel

from mlops_forecast.serving.quantile_lgbm import (
    QuantileLGBM,
    adaptive_conformal_offsets,
)

logger = logging.getLogger(__name__)

_DEFAULT_MLFLOW_URI = Path("mlruns").resolve().as_uri()


def generate_predictions(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_selection_results: pd.DataFrame,  # noqa: ARG001  — DAG ordering only
    params: dict,
) -> pd.DataFrame:
    """Load the Production model from the registry and score X_test.

    The ``model_selection_results`` argument is unused at runtime; it is
    declared as a Kedro input so that the DAG scheduler will not run this
    node until ``model_selection`` has promoted a model to Production.

    Returns a DataFrame with actuals, point predictions, and the calibrated
    lower and upper bounds. When the registered model is the conformalised
    QuantileLGBM wrapper, we additionally compute and log an ACI baseline
    over the same test set.
    """
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    # Pin the experiment so the ACI baseline run logs alongside training.
    mlflow.set_experiment(params.get("experiment_name", "ElectricityForecast"))

    model_uri = f"models:/{params['model_name']}/{params['model_stage']}"
    logger.info("Loading model: %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    y_pred = model.predict(X_test)

    # The QuantileLGBM wrapper returns a 3-column DataFrame; a plain LightGBM
    # registration returns a 1-D array. We support both so this node still
    # works if `prediction_intervals.enabled` is turned off in parameters.yml.
    if isinstance(y_pred, pd.DataFrame):
        load_mw_predicted = y_pred["load_mw_predicted"].values
        lower_bound = y_pred.get("lower_bound", pd.Series(dtype=float)).values
        upper_bound = y_pred.get("upper_bound", pd.Series(dtype=float)).values
    else:
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        load_mw_predicted = y_pred.flatten()
        lower_bound = upper_bound = None

    result = pd.DataFrame(
        {
            "timestamp": X_test.index,
            "load_mw_actual": y_test["load_mw"].values,
            "load_mw_predicted": load_mw_predicted,
        }
    )

    if lower_bound is not None and len(lower_bound):
        result["lower_bound"] = lower_bound
        result["upper_bound"] = upper_bound
        coverage = float(
            (
                (result["load_mw_actual"] >= result["lower_bound"])
                & (result["load_mw_actual"] <= result["upper_bound"])
            ).mean()
        )
        logger.info("Test prediction interval coverage (split-conformal): %.1f%%", coverage * 100)
        _maybe_log_aci_baseline(model, X_test, y_test, result, params)

    result["abs_error"] = (result["load_mw_actual"] - result["load_mw_predicted"]).abs()
    result["mape_contribution"] = result["abs_error"] / result["load_mw_actual"].abs() * 100
    logger.info("Test MAPE: %.2f%%", result["mape_contribution"].mean())
    return result


def _maybe_log_aci_baseline(
    model: PyFuncModel,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    result: pd.DataFrame,
    params: dict,
) -> None:
    """Run ACI over the test set and write columns + MLflow metrics.

    Skips silently if the loaded model isn't a QuantileLGBM (e.g. when
    `prediction_intervals.enabled` is off and the registered artefact is a
    bare LightGBM). The test set must be chronologically sorted; the Kedro
    split node already guarantees that.
    """
    inner = getattr(getattr(model, "_model_impl", None), "python_model", None)
    if not isinstance(inner, QuantileLGBM):
        return

    aci_p = params.get("aci", {})
    if not aci_p.get("enabled", True):
        return

    target_cov = aci_p.get("target_coverage", 0.8)
    gamma = aci_p.get("gamma", 0.005)

    X_ordered = X_test[inner.feature_names]
    y_ordered = y_test["load_mw"].to_numpy()
    raw_lower = inner.lower_model.predict(X_ordered)
    raw_upper = inner.upper_model.predict(X_ordered)

    q_history, hits = adaptive_conformal_offsets(
        y_true=y_ordered,
        y_lower=raw_lower,
        y_upper=raw_upper,
        target_coverage=target_cov,
        gamma=gamma,
        initial_q=inner.conformal_q,
    )

    aci_lower = raw_lower - q_history
    aci_upper = raw_upper + q_history
    result["lower_bound_aci"] = aci_lower
    result["upper_bound_aci"] = aci_upper

    aci_cov = float(hits.mean())
    aci_width = float((aci_upper - aci_lower).mean())
    logger.info(
        "Test prediction interval coverage (ACI, gamma=%.3f): %.1f%% (avg width %.0f MW)",
        gamma,
        aci_cov * 100,
        aci_width,
    )

    # Log alongside the main run so MLflow shows both calibration schemes side by side.
    try:
        with mlflow.start_run(run_name="aci_baseline", nested=False):
            mlflow.log_param("aci_gamma", gamma)
            mlflow.log_param("aci_target_coverage", target_cov)
            mlflow.log_metric("test_aci_coverage", aci_cov)
            mlflow.log_metric("test_aci_avg_width", aci_width)
            mlflow.log_metric("test_aci_q_initial", float(q_history[0]))
            mlflow.log_metric("test_aci_q_final", float(q_history[-1]))
            mlflow.log_metric("test_aci_q_max", float(np.max(q_history)))
    except Exception as exc:  # noqa: BLE001
        # The ACI metrics are diagnostic; don't fail the pipeline if MLflow
        # can't log them (e.g. tracking server temporarily unreachable).
        logger.warning("ACI baseline metrics not logged to MLflow: %s", exc)
