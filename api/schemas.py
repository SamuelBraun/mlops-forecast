"""Pydantic request/response schemas for the FastAPI serving layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """POST /predict. Request payload."""

    start_timestamp: datetime = Field(
        ...,
        description="First timestamp to forecast (UTC ISO-8601)",
        examples=["2020-06-01T00:00:00Z"],
    )
    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of hours to forecast (1–168)",
    )
    wind_mw: list[float] | None = Field(
        default=None,
        description="Optional exogenous wind generation values (must match horizon_hours length)",
    )
    solar_mw: list[float] | None = Field(
        default=None,
        description="Optional exogenous solar generation values",
    )

    @field_validator("wind_mw", "solar_mw", mode="before")
    @classmethod
    def check_length(cls, v: list[float] | None, info: object) -> list[float] | None:  # noqa: ARG003
        return v


class ForecastPoint(BaseModel):
    """Single hourly forecast entry."""

    timestamp: datetime
    load_mw_predicted: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class PredictResponse(BaseModel):
    """POST /predict. Response payload."""

    model_name: str
    model_version: str
    model_stage: str
    forecast: list[ForecastPoint]
    generated_at: datetime


class ModelInfoResponse(BaseModel):
    """GET /model/info. Response payload."""

    model_name: str
    model_version: str
    model_stage: str
    run_id: str
    registered_at: str
    metrics: dict[str, float]
    flavors: list[str]
