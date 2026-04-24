"""Data cleaning nodes: timestamp normalisation, gap filling, outlier handling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Mapping from OPSD column names to internal project names
_COLUMN_MAP = {
    "utc_timestamp": "timestamp",
    "DE_load_actual_entsoe_transparency": "load_mw",
    "DE_wind_generation_actual": "wind_mw",
    "DE_solar_generation_actual": "solar_mw",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename OPSD columns to internal project schema and cast types.

    Args:
        df: Raw OPSD DataFrame.

    Returns:
        DataFrame with renamed columns and UTC-aware DatetimeIndex.
    """
    df = df.rename(columns=_COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Keep only the three German columns we model. The full OPSD CSV
    # has 200+ European country columns that would cause NaN flooding downstream.
    keep = ["load_mw", "wind_mw", "solar_mw"]
    df = df[[c for c in keep if c in df.columns]]
    df[keep] = df[keep].apply(pd.to_numeric, errors="coerce")
    logger.info("Normalised columns (DE-only): %s", list(df.columns))
    return df


def reindex_to_hourly(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Enforce a complete hourly index, forward-filling short gaps.

    Strategy:
      - Gaps ≤ max_gap_hours: interpolate linearly (preserves plausible values)
      - Gaps  > max_gap_hours: leave as NaN with a warning (data quality issue)

    Args:
        df: DataFrame with a UTC DatetimeIndex (hourly, may have gaps).
        params: `data_cleaning` section from parameters.yml.

    Returns:
        DataFrame with a complete hourly DatetimeIndex.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df_reindexed = df.reindex(full_idx)
    n_missing = df_reindexed.isnull().any(axis=1).sum()
    if n_missing > 0:
        logger.warning("%d missing timestamps found. Interpolating short gaps", n_missing)

    max_gap = params["max_gap_hours"]
    method = params["fill_method"]
    for col in df_reindexed.columns:
        null_mask = df_reindexed[col].isnull()
        # Find consecutive null runs
        null_groups = (null_mask != null_mask.shift()).cumsum()
        group_sizes = null_mask.groupby(null_groups).transform("sum")
        short_gaps = null_mask & (group_sizes <= max_gap)
        long_gaps = null_mask & (group_sizes > max_gap)

        if short_gaps.any():
            df_reindexed.loc[short_gaps, col] = np.nan  # ensure NaN before interpolate
            df_reindexed[col] = df_reindexed[col].interpolate(method=method, limit=max_gap)
        if long_gaps.any():
            logger.warning(
                "Column %s has %d values in gaps > %d h. Left as NaN",
                col,
                long_gaps.sum(),
                max_gap,
            )

    logger.info("Reindexed to %d hourly rows", len(df_reindexed))
    return df_reindexed


def remove_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Detect and neutralise outliers in numeric columns.

    Method is controlled by params["outlier_method"]:
      - "iqr": Inter-quartile range fencing (simple, interpretable)
      - "isolation_forest": Multivariate anomaly detection

    Outliers are replaced with NaN and then interpolated (not dropped) so the
    hourly index remains intact.

    Args:
        df: Cleaned DataFrame with complete hourly index.
        params: `data_cleaning` section from parameters.yml.

    Returns:
        DataFrame with outliers replaced by interpolated values.
    """
    method = params["outlier_method"]
    numeric_cols = ["load_mw", "wind_mw", "solar_mw"]

    if method == "iqr":
        multiplier = params["iqr_multiplier"]
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                logger.info("%s: %d outliers removed via IQR", col, mask.sum())
                df.loc[mask, col] = np.nan

    elif method == "isolation_forest":
        contamination = params["isolation_forest_contamination"]
        clf = IsolationForest(contamination=contamination, random_state=42)
        data = df[numeric_cols].dropna()
        preds = clf.fit_predict(data)
        outlier_idx = data.index[preds == -1]
        logger.info("Isolation Forest: %d outlier rows flagged", len(outlier_idx))
        df.loc[outlier_idx, numeric_cols] = np.nan

    else:
        raise ValueError(f"Unknown outlier_method: {method!r}. Use 'iqr' or 'isolation_forest'.")

    # Re-interpolate NaNs introduced by outlier removal
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=24)
    return df


def enforce_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically impossible negative generation values to zero.

    Solar and wind generation cannot be negative. Negative load values indicate
    feed-in surplus and are valid but should be documented.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with generation columns clipped to ≥ 0.
    """
    for col in ["wind_mw", "solar_mw"]:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            logger.warning("%s: %d negative values clipped to 0", col, n_neg)
            df[col] = df[col].clip(lower=0)
    return df
