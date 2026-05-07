"""End-to-end integration test for the data-prep pipelines.

The unit tests in this directory cover individual nodes. This file chains
the four data-side pipelines together on a synthetic OPSD-shaped frame and
asserts that the outputs are internally consistent: row counts add up,
the chronological boundary holds, the target column is not in the feature
matrix, and the train_data_hash that goes to MLflow is deterministic.

It catches the kind of break that wouldn't show up in node-level tests:
the output schema of one node drifting away from what the next node's
parameter map expects, or a parameter key getting renamed without a
follow-up edit in `parameters.yml`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlops_forecast.pipelines.data_cleaning.nodes import (
    enforce_non_negative,
    normalise_columns,
    reindex_to_hourly,
    remove_outliers,
)
from mlops_forecast.pipelines.data_feat_engineering.nodes import build_feature_matrix
from mlops_forecast.pipelines.data_split.nodes import temporal_split


@pytest.fixture(scope="module")
def opsd_like_frame() -> pd.DataFrame:
    """Eight weeks of synthetic hourly OPSD data with mild seasonality."""
    rng = np.random.default_rng(42)
    n_hours = 24 * 7 * 8  # 1,344 rows
    idx = pd.date_range("2019-01-01", periods=n_hours, freq="h", tz="UTC")
    # Daily + weekly cosines plus noise; range chosen so IQR/range checks pass.
    t = np.arange(n_hours)
    daily = 10_000 * np.cos(2 * np.pi * t / 24)
    weekly = 5_000 * np.cos(2 * np.pi * t / 168)
    load = 55_000 + daily + weekly + rng.normal(0, 1500, n_hours)
    wind = rng.uniform(0, 18_000, n_hours)
    solar = np.where((idx.hour >= 7) & (idx.hour <= 19), rng.uniform(0, 14_000, n_hours), 0.0)
    return pd.DataFrame(
        {
            "utc_timestamp": idx,
            "DE_load_actual_entsoe_transparency": load,
            "DE_wind_generation_actual": wind,
            "DE_solar_generation_actual": solar,
        }
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)
    df = reindex_to_hourly(df, {"fill_method": "time", "max_gap_hours": 6})
    df = remove_outliers(
        df,
        {
            "outlier_method": "iqr",
            "iqr_multiplier": 3.0,
            "isolation_forest_contamination": 0.01,
        },
    )
    return enforce_non_negative(df)


def _feat_params() -> dict:
    return {
        "lag_hours": [1, 24, 168],
        "rolling_windows": [6, 24],
        "country_holidays": "DE",
        "fourier_periods": [24, 168],
        "fourier_order": 2,
        "drop_na_after_features": True,
    }


def _split_params(df: pd.DataFrame) -> dict:
    # Carve a chronological 70/15/15 from whatever rows survived feature
    # engineering. Computing the cutoffs from the actual frame keeps the
    # test independent of the synthetic generator's exact output length.
    n = len(df)
    train_end = df.index[int(n * 0.70)]
    val_end = df.index[int(n * 0.85)]
    return {
        "train_end": str(train_end),
        "val_end": str(val_end),
    }


def test_data_prep_pipeline_runs_end_to_end(opsd_like_frame: pd.DataFrame) -> None:
    cleaned = _clean(opsd_like_frame)
    features = build_feature_matrix(cleaned, _feat_params())

    assert len(features) > 0
    assert "load_mw" in features.columns
    assert features.isna().sum().sum() == 0

    X_train, y_train, X_val, y_val, X_test, y_test, meta_json = temporal_split(
        features, _split_params(features)
    )

    # Every row of the feature matrix lands in exactly one of the three splits.
    assert len(X_train) + len(X_val) + len(X_test) == len(features)

    # Strict chronological boundaries.
    assert X_train.index.max() < X_val.index.min()
    assert X_val.index.max() < X_test.index.min()

    # The target is split out, not duplicated into X.
    assert "load_mw" not in X_train.columns
    assert "load_mw" in y_train.columns

    meta = json.loads(meta_json)
    assert meta["n_train"] == len(X_train)
    assert meta["n_val"] == len(X_val)
    assert meta["n_test"] == len(X_test)


def test_train_data_hash_is_deterministic(opsd_like_frame: pd.DataFrame) -> None:
    """Same input twice produces the same train_data_hash.

    This is the property MLflow uses to detect silent input changes between
    runs; if the hash isn't stable, the diff view becomes useless.
    """
    f = build_feature_matrix(_clean(opsd_like_frame.copy()), _feat_params())
    _, _, _, _, _, _, meta_a = temporal_split(f, _split_params(f))
    _, _, _, _, _, _, meta_b = temporal_split(
        build_feature_matrix(_clean(opsd_like_frame.copy()), _feat_params()),
        _split_params(f),
    )
    assert json.loads(meta_a)["train_data_hash"] == json.loads(meta_b)["train_data_hash"]


def test_no_temporal_leakage_in_lag_features(opsd_like_frame: pd.DataFrame) -> None:
    """A row's lag_24h must equal load_mw 24 rows earlier in the cleaned frame.

    The lag feature is the place where leakage tends to creep in (e.g. if
    someone replaces `shift(h)` with `shift(-h)` by accident). This test
    catches that without needing to inspect the wider matrix.
    """
    cleaned = _clean(opsd_like_frame)
    features = build_feature_matrix(cleaned, _feat_params())
    # Pick a row well past the lag horizon and the dropped-NaN prefix.
    sample = features.iloc[200]
    expected = cleaned.loc[sample.name - pd.Timedelta(hours=24), "load_mw"]
    assert sample["lag_24h"] == pytest.approx(expected, rel=1e-9)
