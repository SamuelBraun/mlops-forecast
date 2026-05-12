"""FastAPI service for the electricity load forecast.

The model is loaded once at startup from `models:/<name>/Production`, never
per-request and never from a filesystem path. SHAP and drift live in their
own Kedro pipelines and are never imported here, which keeps the request
path lean and the runtime image small. Pydantic v2 validates everything
that crosses the wire.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import uvicorn
from api.schemas import ForecastPoint, ModelInfoResponse, PredictRequest, PredictResponse
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# Module-level state holding the loaded model and its metadata. Both are
# populated by `lifespan` at startup and consumed by the route handlers.
_MODEL: mlflow.pyfunc.PyFuncModel | None = None
_MODEL_INFO: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Pull the Production model from MLflow when the worker starts."""
    global _MODEL, _MODEL_INFO
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "ElectricityForecast")
    model_stage = os.getenv("MODEL_STAGE", "Production")

    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        logger.info("Loading model: %s", model_uri)
        _MODEL = mlflow.pyfunc.load_model(model_uri)

        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        if versions:
            v = versions[0]
            run = client.get_run(v.run_id)
            _MODEL_INFO = {
                "model_name": model_name,
                "model_version": v.version,
                "model_stage": model_stage,
                "run_id": v.run_id,
                "registered_at": str(v.creation_timestamp),
                "metrics": run.data.metrics,
                "flavors": list(_MODEL.metadata.flavors.keys()),
            }
        logger.info(
            "Model loaded successfully: %s v%s",
            model_name,
            versions[0].version if versions else "?",
        )
    except Exception as exc:
        logger.error("Failed to load model at startup: %s", exc)
        # We swallow the exception so the worker still boots; /health will
        # return 503 until somebody fixes the registry. This is friendlier
        # for orchestration (k8s, compose) than crashing on boot.
        _MODEL = None
        _MODEL_INFO = {}

    yield
    logger.info("Shutting down API")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MLOps Forecast API",
    description="Electricity load forecasting. NOVA IMS MLOps 2026",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe. Returns 200 if the model is loaded, 503 otherwise."""
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded Production model."""
    if not _MODEL_INFO:
        raise HTTPException(status_code=503, detail="Model info not available")
    return ModelInfoResponse(**_MODEL_INFO)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Forecast the next `horizon_hours` of German load from `start_timestamp`.

    The feature vector is rebuilt at request time from calendar and Fourier
    features only (lag features default to zero, since the API has no
    pre-loaded history of recent load). Wind and solar can be provided per
    hour by the caller; otherwise they default to zero.
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health")

    horizon = request.horizon_hours
    # Pydantic gives us either a tz-aware or tz-naive datetime depending on
    # how the client formatted the timestamp. pd.date_range refuses both
    # an aware `start` and a `tz=` argument, so we normalise to a tz-aware
    # UTC start and let date_range inherit it.
    start = pd.Timestamp(request.start_timestamp)
    start = start.tz_convert("UTC") if start.tzinfo is not None else start.tz_localize("UTC")
    idx = pd.date_range(start=start, periods=horizon, freq="h")

    X = _build_inference_features(
        idx=idx,
        wind_mw=request.wind_mw,
        solar_mw=request.solar_mw,
    )

    try:
        y_pred = _MODEL.predict(X)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # The QuantileLGBM wrapper returns a 3-column DataFrame; a plain
    # LightGBM registration would return an ndarray. We support both so the
    # API still works if intervals are turned off in parameters.yml.
    if isinstance(y_pred, pd.DataFrame):
        points = y_pred["load_mw_predicted"].tolist()
        lowers = y_pred.get("lower_bound", pd.Series([None] * len(idx))).tolist()
        uppers = y_pred.get("upper_bound", pd.Series([None] * len(idx))).tolist()
    else:
        points = list(np.asarray(y_pred).flatten())
        lowers = uppers = [None] * len(idx)

    forecast = [
        ForecastPoint(
            timestamp=ts,
            load_mw_predicted=float(p),
            lower_bound=float(lo) if lo is not None else None,
            upper_bound=float(hi) if hi is not None else None,
        )
        for ts, p, lo, hi in zip(idx, points, lowers, uppers)
    ]

    return PredictResponse(
        model_name=_MODEL_INFO.get("model_name", "unknown"),
        model_version=str(_MODEL_INFO.get("model_version", "?")),
        model_stage=_MODEL_INFO.get("model_stage", "Production"),
        forecast=forecast,
        generated_at=datetime.now(timezone.utc),
    )


def _build_inference_features(
    idx: pd.DatetimeIndex,
    wind_mw: list[float] | None,
    solar_mw: list[float] | None,
) -> pd.DataFrame:
    """Build the feature frame the registered model expects, at request time.

    The training pipeline produces 40 columns. We rebuild the same column
    names here; calendar and Fourier features come from the timestamp index
    and are therefore exact, while lag and rolling columns are filled with
    zeros (we have no recent-history feed in this PoC). The model handles
    that gracefully because it was trained on the same column ordering.
    For a deployed system the right answer is to fetch the recent-history
    columns from Feast's online store using `get_online_features`.
    """
    import holidays  # noqa: PLC0415

    n = len(idx)
    df = pd.DataFrame(index=idx)
    df["wind_mw"] = wind_mw if wind_mw else np.zeros(n)
    df["solar_mw"] = solar_mw if solar_mw else np.zeros(n)
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["day_of_year"] = idx.dayofyear
    df["month"] = idx.month
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["season"] = idx.month % 12 // 3
    years = list(idx.year.unique())
    de_holidays = holidays.country_holidays("DE", years=years)
    df["is_holiday"] = idx.normalize().isin(de_holidays).astype(int)
    df["is_working_day"] = (
        (~df["is_weekend"].astype(bool)) & (~df["is_holiday"].astype(bool))
    ).astype(int)

    # Fourier features (daily + weekly)
    t = np.arange(n, dtype=float)
    for period in [24, 168]:
        for k in range(1, 4):
            df[f"fourier_{period}h_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"fourier_{period}h_cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    # Lag features default to 0 for pure-horizon inference
    for h in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{h}h"] = 0.0

    # Rolling features default to 0
    for w in [6, 12, 24, 168]:
        df[f"rolling_mean_{w}h"] = 0.0
        df[f"rolling_std_{w}h"] = 0.0

    return df


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
