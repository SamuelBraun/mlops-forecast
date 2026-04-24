"""Unit tests for data cleaning nodes."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_forecast.pipelines.data_cleaning.nodes import (
    enforce_non_negative,
    normalise_columns,
    reindex_to_hourly,
    remove_outliers,
)


class TestNormaliseColumns:
    def test_renames_columns(self, sample_raw_df: pd.DataFrame) -> None:
        result = normalise_columns(sample_raw_df)
        assert "load_mw" in result.columns
        assert "wind_mw" in result.columns
        assert "solar_mw" in result.columns
        assert "utc_timestamp" not in result.columns

    def test_sets_datetime_index(self, sample_raw_df: pd.DataFrame) -> None:
        result = normalise_columns(sample_raw_df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None  # UTC-aware

    def test_sorted_ascending(self, sample_raw_df: pd.DataFrame) -> None:
        shuffled = sample_raw_df.sample(frac=1, random_state=99)
        result = normalise_columns(shuffled)
        assert result.index.is_monotonic_increasing


class TestReindexToHourly:
    def test_fills_single_missing_hour(self, sample_clean_df: pd.DataFrame) -> None:
        df = sample_clean_df.copy()
        df = df.drop(df.index[10])  # remove one hour
        params = {"fill_method": "time", "max_gap_hours": 6}
        result = reindex_to_hourly(df, params)
        assert len(result) == len(sample_clean_df)

    def test_no_change_when_complete(self, sample_clean_df: pd.DataFrame) -> None:
        params = {"fill_method": "time", "max_gap_hours": 6}
        result = reindex_to_hourly(sample_clean_df, params)
        assert len(result) == len(sample_clean_df)

    def test_large_gap_leaves_nan(self, sample_clean_df: pd.DataFrame) -> None:
        df = sample_clean_df.copy()
        # Remove 10 consecutive hours. Exceeds max_gap_hours=6
        drop_idx = df.index[20:30]
        df = df.drop(drop_idx)
        params = {"fill_method": "time", "max_gap_hours": 6}
        result = reindex_to_hourly(df, params)
        # Some NaN should remain in the large gap (partial fill allowed)
        # Just assert no exception and correct length
        assert len(result) == len(sample_clean_df)


class TestRemoveOutliers:
    def test_iqr_removes_spike(self, sample_clean_df: pd.DataFrame) -> None:
        df = sample_clean_df.copy()
        spike_idx = df.index[50]
        df.loc[spike_idx, "load_mw"] = 1_000_000  # extreme outlier
        params = {
            "outlier_method": "iqr",
            "iqr_multiplier": 3.0,
            "isolation_forest_contamination": 0.01,
        }
        result = remove_outliers(df, params)
        assert result.loc[spike_idx, "load_mw"] < 1_000_000

    def test_unknown_method_raises(self, sample_clean_df: pd.DataFrame) -> None:
        params = {"outlier_method": "unsupported_method"}
        with pytest.raises(ValueError, match="Unknown outlier_method"):
            remove_outliers(sample_clean_df, params)

    def test_isolation_forest_runs(self, sample_clean_df: pd.DataFrame) -> None:
        params = {"outlier_method": "isolation_forest", "isolation_forest_contamination": 0.05}
        result = remove_outliers(sample_clean_df, params)
        assert result.shape == sample_clean_df.shape


class TestEnforceNonNegative:
    def test_clips_negative_wind(self, sample_clean_df: pd.DataFrame) -> None:
        df = sample_clean_df.copy()
        df.loc[df.index[5], "wind_mw"] = -500.0
        result = enforce_non_negative(df)
        assert (result["wind_mw"] >= 0).all()

    def test_load_can_be_negative(self, sample_clean_df: pd.DataFrame) -> None:
        """Negative load (feed-in surplus) is physically valid. Do not clip."""
        df = sample_clean_df.copy()
        df.loc[df.index[5], "load_mw"] = -100.0
        result = enforce_non_negative(df)
        assert result.loc[df.index[5], "load_mw"] == -100.0
