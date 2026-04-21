"""Root conftest: session-scoped fixtures shared across all test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_raw_df() -> pd.DataFrame:
    """Minimal OPSD-shaped dataframe for pipeline unit tests."""
    rng = np.random.default_rng(42)
    n = 24 * 14  # two weeks of hourly data
    idx = pd.date_range("2019-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "utc_timestamp": idx,
            "DE_load_actual_entsoe_transparency": rng.normal(55_000, 5_000, n).clip(30_000, 80_000),
            "DE_wind_generation_actual": rng.uniform(0, 20_000, n),
            "DE_solar_generation_actual": np.where(
                (idx.hour >= 7) & (idx.hour <= 19),
                rng.uniform(0, 15_000, len(idx)),
                0.0,
            ),
        }
    )


@pytest.fixture(scope="session")
def sample_clean_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned version of sample_raw_df with renamed columns and sorted index."""
    df = sample_raw_df.copy()
    df = df.rename(
        columns={
            "utc_timestamp": "timestamp",
            "DE_load_actual_entsoe_transparency": "load_mw",
            "DE_wind_generation_actual": "wind_mw",
            "DE_solar_generation_actual": "solar_mw",
        }
    )
    df = df.set_index("timestamp").sort_index()
    return df
