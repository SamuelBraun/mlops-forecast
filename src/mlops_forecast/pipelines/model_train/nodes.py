"""Model training: Seasonal Naive baseline, Prophet, LightGBM.

All runs logged to MLflow with parameters, metrics, and artifacts.
SHAP values saved for the LightGBM model.

The LightGBM trainer also fits two quantile-regression heads (alpha=0.1, 0.9)
and ships all three models inside ``mlops_forecast.serving.QuantileLGBM`` so
``model.predict(X)`` returns a DataFrame with columns ``[load_mw_predicted,
lower_bound, upper_bound]``. The wrapper class lives in the lean ``serving``
package, not here. The API container deserialises it without importing Kedro.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow.pyfunc import PythonModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlops_forecast.serving.quantile_lgbm import QuantileLGBM, conformal_offset

logger = logging.getLogger(__name__)

# File-based MLflow store under `./mlruns`, matching the MLflow 2.x default.
# We set this explicitly (rather than relying on the default) so the path is
# absolute and the same regardless of which directory `kedro run` is invoked
# from. Override with the `MLFLOW_TRACKING_URI` env var if the team is using
# a remote tracking server.
_DEFAULT_MLFLOW_URI = f"file://{Path('mlruns').resolve()}"


def _setup_mlflow(experiment_name: str) -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)
    mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mape": _mape(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Seasonal Naive
# ---------------------------------------------------------------------------


def train_seasonal_naive(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: dict[str, Any],
    split_metadata: str,
) -> dict[str, Any]:
    """Seasonal naive at T-168h: predict y_t = y_{t-168}.

    The strict "same hour last week" baseline. For each validation hour t,
    the prediction is the actual load 168 hours earlier. The first 168 hours
    of validation use the tail of the training partition (so we never look
    forward in time). This is the textbook seasonal-naive used in load
    forecasting and the falsifiability floor for the production model.
    """
    season_period = params["seasonal_naive"]["seasonality_period"]
    meta = json.loads(split_metadata)
    _setup_mlflow(params["experiment_name"])

    with mlflow.start_run(run_name="seasonal_naive") as run:
        mlflow.log_params(
            {
                "model_type": "seasonal_naive",
                "seasonality_period": season_period,
                "train_data_hash": meta["train_data_hash"],
            }
        )

        # y_pred[t] = actual load 168h before t. The first 168 entries
        # come from the tail of training; the rest are y_val shifted by
        # season_period, which is the proper y_t = y_{t-168} predictor.
        train_tail = y_train["load_mw"].iloc[-season_period:].values
        val_arr = y_val["load_mw"].values
        if len(val_arr) <= season_period:
            y_pred = train_tail[: len(val_arr)]
        else:
            y_pred = np.concatenate([train_tail, val_arr[:-season_period]])
        metrics = _compute_metrics(val_arr, y_pred)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        logger.info("Seasonal Naive val_mape=%.2f%%", metrics["mape"])

        # Register as a pyfunc model. The deployed model needs a recent
        # history window to predict from; we ship the last 168 hours of
        # training as the bootstrap and let the caller supply later history
        # in the input DataFrame's `load_mw` column if available.
        class _NaiveModel(PythonModel):
            def __init__(self, bootstrap_: np.ndarray, period_: int) -> None:
                self.bootstrap_ = bootstrap_
                self.period_ = period_

            def predict(  # noqa: ARG002
                self,
                context: Any,
                model_input: pd.DataFrame,
                params: dict[str, Any] | None = None,
            ) -> np.ndarray:
                n = len(model_input)
                if "load_mw" in model_input.columns:
                    arr = model_input["load_mw"].values
                    if n <= self.period_:
                        out = self.bootstrap_[:n]
                    else:
                        out = np.concatenate([self.bootstrap_, arr[: n - self.period_]])
                else:
                    out = np.tile(self.bootstrap_, int(np.ceil(n / self.period_)))[:n]
                return out

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_NaiveModel(train_tail, season_period),
            registered_model_name=params["register_model_name"],
        )
        run_id = run.info.run_id

    return {
        "model_type": "seasonal_naive",
        "run_id": run_id,
        **{f"val_{k}": v for k, v in metrics.items()},
    }


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: dict[str, Any],
    split_metadata: str,
) -> tuple[dict[str, Any], pd.DataFrame, plt.Figure]:
    """LightGBM with early stopping and SHAP values.

    Returns:
        (run_info_dict, shap_values_df, shap_summary_figure)
    """
    lgbm_p = params["lightgbm"]
    cv_p = params.get("walk_forward_cv", {"enabled": False})
    meta = json.loads(split_metadata)
    _setup_mlflow(params["experiment_name"])

    def _make_lgbm(objective: str = "regression", alpha: float | None = None) -> lgb.LGBMRegressor:
        kwargs = dict(
            n_estimators=lgbm_p["n_estimators"],
            learning_rate=lgbm_p["learning_rate"],
            num_leaves=lgbm_p["num_leaves"],
            min_child_samples=lgbm_p["min_child_samples"],
            subsample=lgbm_p["subsample"],
            colsample_bytree=lgbm_p["colsample_bytree"],
            reg_alpha=lgbm_p["reg_alpha"],
            reg_lambda=lgbm_p["reg_lambda"],
            random_state=lgbm_p["random_state"],
            n_jobs=-1,
            verbose=-1,
            objective=objective,
        )
        if alpha is not None:
            kwargs["alpha"] = alpha
        return lgb.LGBMRegressor(**kwargs)

    with mlflow.start_run(run_name="lightgbm") as run:
        mlflow.log_params(
            {"model_type": "lightgbm", "train_data_hash": meta["train_data_hash"], **lgbm_p}
        )

        # ------------------------------------------------------------------
        # Walk-forward CV (expanding window). Logs each fold as a nested run
        # so MLflow shows the variance across time, not just one number.
        # ------------------------------------------------------------------
        if cv_p.get("enabled", False):
            n_splits = cv_p["n_splits"]
            fold_size = len(X_train) // (n_splits + 1)
            fold_mapes: list[float] = []
            for i in range(n_splits):
                fold_train_end = fold_size * (i + 1)
                fold_val_end = fold_size * (i + 2)
                Xt = X_train.iloc[:fold_train_end]
                yt = y_train["load_mw"].iloc[:fold_train_end]
                Xv = X_train.iloc[fold_train_end:fold_val_end]
                yv = y_train["load_mw"].iloc[fold_train_end:fold_val_end]
                with mlflow.start_run(run_name=f"fold_{i + 1}", nested=True):
                    fm = _make_lgbm()
                    fm.fit(
                        Xt,
                        yt,
                        eval_set=[(Xv, yv)],
                        callbacks=[
                            lgb.early_stopping(lgbm_p["early_stopping_rounds"], verbose=False)
                        ],
                    )
                    fold_metrics = _compute_metrics(yv.values, fm.predict(Xv))
                    fold_mapes.append(fold_metrics["mape"])
                    mlflow.log_metrics({f"fold_{k}": v for k, v in fold_metrics.items()})
                    mlflow.log_param("fold_index", i + 1)
            mlflow.log_metric("cv_mape_mean", float(np.mean(fold_mapes)))
            mlflow.log_metric("cv_mape_std", float(np.std(fold_mapes)))
            logger.info(
                "Walk-forward CV: mean MAPE=%.2f%% (±%.2f%%) across %d folds",
                float(np.mean(fold_mapes)),
                float(np.std(fold_mapes)),
                n_splits,
            )

        # ------------------------------------------------------------------
        # Final fit on the full train set
        # ------------------------------------------------------------------
        model = _make_lgbm()
        model.fit(
            X_train,
            y_train["load_mw"],
            eval_set=[(X_val, y_val["load_mw"])],
            callbacks=[
                lgb.early_stopping(lgbm_p["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        y_pred = model.predict(X_val)
        metrics = _compute_metrics(y_val["load_mw"].values, y_pred)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        best_iter = model.best_iteration_ or lgbm_p["n_estimators"]
        mlflow.log_param("best_iteration", best_iter)
        logger.info("LightGBM val_mape=%.2f%%, best_iter=%d", metrics["mape"], best_iter)

        # ------------------------------------------------------------------
        # Quantile heads for prediction intervals
        # ------------------------------------------------------------------
        intervals_p = params.get("prediction_intervals", {"enabled": False})
        lower_alpha = intervals_p.get("lower_alpha", 0.1)
        upper_alpha = intervals_p.get("upper_alpha", 0.9)
        lower_model: lgb.LGBMRegressor | None = None
        upper_model: lgb.LGBMRegressor | None = None
        conformal_q = 0.0
        if intervals_p.get("enabled", False):
            logger.info(
                "Fitting quantile heads (alpha=%.2f / %.2f) for prediction intervals",
                lower_alpha,
                upper_alpha,
            )
            lower_model = _make_lgbm(objective="quantile", alpha=lower_alpha)
            upper_model = _make_lgbm(objective="quantile", alpha=upper_alpha)
            lower_model.fit(
                X_train,
                y_train["load_mw"],
                eval_set=[(X_val, y_val["load_mw"])],
                callbacks=[lgb.early_stopping(lgbm_p["early_stopping_rounds"], verbose=False)],
            )
            upper_model.fit(
                X_train,
                y_train["load_mw"],
                eval_set=[(X_val, y_val["load_mw"])],
                callbacks=[lgb.early_stopping(lgbm_p["early_stopping_rounds"], verbose=False)],
            )
            # Raw quantile-head coverage on validation set, before calibration.
            lo_raw = lower_model.predict(X_val)
            hi_raw = upper_model.predict(X_val)
            y_val_arr = y_val["load_mw"].values
            raw_cov = float(((y_val_arr >= lo_raw) & (y_val_arr <= hi_raw)).mean())
            mlflow.log_metric("val_pi_coverage_raw", raw_cov)
            mlflow.log_metric("val_pi_avg_width_raw", float((hi_raw - lo_raw).mean()))

            # Split-conformal calibration on the same val set. We use it as the
            # calibration set rather than carving off a separate split because
            # (a) val is already disjoint from train, and (b) the OPSD dataset
            # is small enough that a third partition would reduce calibration
            # sample size below the recommended n>=200 for finite-sample CQR.
            # In a production rerun with more data we would add a held-out
            # calibration window between val and test.
            target_cov = upper_alpha - lower_alpha
            miscoverage = 1.0 - target_cov
            conformal_q = conformal_offset(y_val_arr, lo_raw, hi_raw, miscoverage)
            mlflow.log_metric("conformal_q_mw", conformal_q)
            lo_cal = lo_raw - conformal_q
            hi_cal = hi_raw + conformal_q
            cal_cov = float(((y_val_arr >= lo_cal) & (y_val_arr <= hi_cal)).mean())
            mlflow.log_metric("val_pi_coverage", cal_cov)
            mlflow.log_metric("val_pi_avg_width", float((hi_cal - lo_cal).mean()))
            logger.info(
                "Prediction intervals: raw coverage %.1f%%, "
                "conformal q=%.0f MW, calibrated coverage %.1f%% (target %.0f%%)",
                raw_cov * 100,
                conformal_q,
                cal_cov * 100,
                target_cov * 100,
            )

        # Feature importance
        fi_df = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fi_path = Path("data/06_models/feature_importance.csv")
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        # SHAP
        explainer = shap.TreeExplainer(model)
        n_shap = min(300, len(X_val))
        shap_sample = X_val.sample(n_shap, random_state=42)
        shap_values_obj = explainer(shap_sample)
        shap_df = pd.DataFrame(
            shap_values_obj.values, columns=X_val.columns, index=shap_sample.index
        )

        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values_obj,
            shap_sample,
            show=False,
            max_display=params["shap"]["max_display"],
        )
        plt.tight_layout()
        mlflow.log_figure(fig, "shap_summary.png")

        # ------------------------------------------------------------------
        # Register either the bare LightGBM (no intervals) or the quantile
        # bundle (point + lower + upper). The bundle ships as a pyfunc so the
        # API can call .predict() and get a 3-column DataFrame back.
        # ------------------------------------------------------------------
        if lower_model is not None and upper_model is not None:
            wrapper = QuantileLGBM(
                point_model=model,
                lower_model=lower_model,
                upper_model=upper_model,
                feature_names=list(X_train.columns),
                conformal_q=conformal_q,
            )
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapper,
                registered_model_name=params["register_model_name"],
            )
        else:
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=params["register_model_name"],
            )
        run_id = run.info.run_id

    return (
        {
            "model_type": "lightgbm",
            "run_id": run_id,
            **{f"val_{k}": v for k, v in metrics.items()},
            "best_iteration": best_iter,
        },
        shap_df,
        fig,
    )


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------


def train_prophet(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: dict[str, Any],
    split_metadata: str,
) -> dict[str, Any]:
    """Prophet model with additive/multiplicative seasonality."""
    from prophet import Prophet  # noqa: PLC0415

    pp = params["prophet"]
    meta = json.loads(split_metadata)
    _setup_mlflow(params["experiment_name"])

    with mlflow.start_run(run_name="prophet") as run:
        mlflow.log_params(
            {"model_type": "prophet", "train_data_hash": meta["train_data_hash"], **pp}
        )

        train_df = pd.DataFrame(
            {
                "ds": y_train.index.tz_localize(None),
                "y": y_train["load_mw"].values,
            }
        )
        model = Prophet(
            seasonality_mode=pp["seasonality_mode"],
            daily_seasonality=pp["daily_seasonality"],
            weekly_seasonality=pp["weekly_seasonality"],
            yearly_seasonality=pp["yearly_seasonality"],
            changepoint_prior_scale=pp["changepoint_prior_scale"],
            n_changepoints=pp["n_changepoints"],
        )
        import logging as _logging  # noqa: PLC0415

        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        model.fit(train_df)

        future_df = pd.DataFrame({"ds": y_val.index.tz_localize(None)})
        forecast = model.predict(future_df)
        y_pred = forecast["yhat"].values
        metrics = _compute_metrics(y_val["load_mw"].values, y_pred)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        logger.info("Prophet val_mape=%.2f%%", metrics["mape"])

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=params["register_model_name"],
        )
        run_id = run.info.run_id

    return {
        "model_type": "prophet",
        "run_id": run_id,
        **{f"val_{k}": v for k, v in metrics.items()},
    }
