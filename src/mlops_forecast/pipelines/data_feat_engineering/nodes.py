"""Feature engineering for the electricity load model.

The leakage rule we follow throughout: a feature for time t may only use
values from times <= t-1. Lags are produced with `.shift(h)` for h >= 1,
and rolling statistics use `.shift(1)` before the rolling window so that
the value at t is excluded from its own summary. We never fit anything on
the full series before splitting; the training pipeline is responsible for
respecting the train/val/test boundary it gets handed.
"""

from __future__ import annotations

import logging
from typing import Any

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, lag_hours: list[int]) -> pd.DataFrame:
    """Add lag features: load at t-h for each h in lag_hours.

    Args:
        df: DataFrame with 'load_mw' column and hourly DatetimeIndex.
        lag_hours: List of lag sizes in hours.

    Returns:
        DataFrame with additional lag_<h>h columns.
    """
    for h in lag_hours:
        df[f"lag_{h}h"] = df["load_mw"].shift(h)
        logger.debug("Created lag feature: lag_%dh", h)
    return df


def create_rolling_features(df: pd.DataFrame, rolling_windows: list[int]) -> pd.DataFrame:
    """Add rolling mean and std features over specified hour windows.

    Uses shift(1) before rolling to avoid including the current observation,
    which would cause look-ahead bias within the window.

    Args:
        df: DataFrame with 'load_mw' column and hourly DatetimeIndex.
        rolling_windows: List of window sizes in hours.

    Returns:
        DataFrame with rolling_mean_<w>h and rolling_std_<w>h columns.
    """
    # The shift(1) is the whole point: a rolling mean over t-w..t would
    # contain the value at t in its own predictor, which is leakage.
    shifted = df["load_mw"].shift(1)
    for w in rolling_windows:
        df[f"rolling_mean_{w}h"] = shifted.rolling(window=w, min_periods=w // 2).mean()
        df[f"rolling_std_{w}h"] = shifted.rolling(window=w, min_periods=w // 2).std()
        logger.debug("Created rolling features for window %dh", w)
    return df


def create_calendar_features(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Add calendar-based features: hour, day-of-week, month, holiday flag, season.

    Args:
        df: DataFrame with UTC DatetimeIndex.
        country_code: ISO 3166-1 alpha-2 code for holiday calendar (e.g. 'DE').

    Returns:
        DataFrame enriched with calendar features.
    """
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek  # 0 = Monday
    df["day_of_year"] = idx.dayofyear
    df["month"] = idx.month
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Season (Northern hemisphere)
    df["season"] = idx.month % 12 // 3  # 0=winter, 1=spring, 2=summer, 3=autumn

    # Public holidays
    years = idx.year.unique().tolist()
    country_holidays = holidays.country_holidays(country_code, years=years)
    df["is_holiday"] = idx.normalize().isin(country_holidays).astype(int)

    # Working day flag (business day + not holiday)
    df["is_working_day"] = (
        (~df["is_weekend"].astype(bool)) & (~df["is_holiday"].astype(bool))
    ).astype(int)

    logger.info("Calendar features added: hour, day_of_week, month, is_holiday, is_working_day")
    return df


def create_fourier_features(df: pd.DataFrame, periods: list[int], order: int) -> pd.DataFrame:
    """Add Fourier terms to capture multi-seasonality.

    Generates sine and cosine pairs at each harmonic up to `order`
    for each period (e.g. 24h daily, 168h weekly).

    Args:
        df: DataFrame with hourly DatetimeIndex.
        periods: List of seasonality periods in hours.
        order: Number of Fourier pairs per period.

    Returns:
        DataFrame with fourier_<period>h_sin_<k> and fourier_<period>h_cos_<k> columns.
    """
    t = np.arange(len(df), dtype=float)
    for period in periods:
        for k in range(1, order + 1):
            df[f"fourier_{period}h_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"fourier_{period}h_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    logger.info("Fourier features added: periods=%s, order=%d", periods, order)
    return df


def build_feature_matrix(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Run every feature step in order and return the full matrix.

    The first ~168 rows pick up NaN from the deepest lag, so we drop them
    here rather than later. Downstream code can assume a clean frame.
    """
    logger.info("Starting feature engineering on %d rows", len(df))
    df = df.copy()

    df = create_lag_features(df, params["lag_hours"])
    df = create_rolling_features(df, params["rolling_windows"])
    df = create_calendar_features(df, params["country_holidays"])
    df = create_fourier_features(df, params["fourier_periods"], params["fourier_order"])

    if params.get("drop_na_after_features", True):
        before = len(df)
        df = df.dropna()
        logger.info("Dropped %d rows with NaN after feature engineering", before - len(df))

    logger.info("Feature matrix: %d rows × %d columns", *df.shape)
    return df
