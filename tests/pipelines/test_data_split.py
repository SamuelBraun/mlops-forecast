"""Unit tests for temporal split node. Chronological integrity is non-negotiable."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlops_forecast.pipelines.data_split.nodes import temporal_split


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """2-year feature matrix with hourly index. Enough for a meaningful split."""
    rng = np.random.default_rng(0)
    n = 24 * 365 * 2  # 2 years hourly
    idx = pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"load_mw": rng.normal(55000, 5000, n)}, index=idx)
    df["lag_1h"] = df["load_mw"].shift(1)
    df["hour"] = idx.hour
    df = df.dropna()
    return df


@pytest.fixture
def split_params() -> dict:
    return {
        "train_end": "2018-12-31 23:00:00",
        "val_end": "2019-06-30 23:00:00",
        "cv_n_splits": 3,
        "cv_gap_hours": 24,
        "cv_test_size_hours": 168,
    }


class TestTemporalSplit:
    def test_train_precedes_val(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = temporal_split(feature_df, split_params)
        assert X_train.index.max() < X_val.index.min()

    def test_val_precedes_test(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = temporal_split(feature_df, split_params)
        assert X_val.index.max() < X_test.index.min()

    def test_no_overlap(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        X_train, _, X_val, _, X_test, _, _ = temporal_split(feature_df, split_params)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_covers_all_data(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        X_train, _, X_val, _, X_test, _, _ = temporal_split(feature_df, split_params)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(feature_df)

    def test_metadata_contains_hash(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        _, _, _, _, _, _, metadata_json = temporal_split(feature_df, split_params)
        meta = json.loads(metadata_json)
        assert "train_data_hash" in meta
        assert len(meta["train_data_hash"]) == 32  # MD5 hex

    def test_metadata_correct_sizes(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        X_train, _, X_val, _, X_test, _, metadata_json = temporal_split(feature_df, split_params)
        meta = json.loads(metadata_json)
        assert meta["n_train"] == len(X_train)
        assert meta["n_val"] == len(X_val)
        assert meta["n_test"] == len(X_test)

    def test_target_not_in_X(self, feature_df: pd.DataFrame, split_params: dict) -> None:
        X_train, y_train, _, _, _, _, _ = temporal_split(feature_df, split_params)
        assert "load_mw" not in X_train.columns
        assert "load_mw" in y_train.columns
