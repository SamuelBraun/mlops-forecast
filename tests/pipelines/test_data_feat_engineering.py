"""Unit tests for feature engineering nodes. Leakage prevention is critical."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_forecast.pipelines.data_feat_engineering.nodes import (
    create_calendar_features,
    create_fourier_features,
    create_lag_features,
    create_rolling_features,
)


class TestLagFeatures:
    def test_creates_expected_columns(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_clean_df.copy(), [1, 24])
        assert "lag_1h" in result.columns
        assert "lag_24h" in result.columns

    def test_lag_1_is_previous_value(self, sample_clean_df: pd.DataFrame) -> None:
        df = sample_clean_df.copy()
        result = create_lag_features(df, [1])
        # At row i, lag_1h should equal load_mw at row i-1
        assert result["lag_1h"].iloc[1] == df["load_mw"].iloc[0]

    def test_no_future_leakage(self, sample_clean_df: pd.DataFrame) -> None:
        """Lag features must only look backward, never forward."""
        df = sample_clean_df.copy()
        result = create_lag_features(df, [1])
        # lag_1h at position i must equal load_mw at position i-1 (not i+1)
        for i in range(1, min(10, len(result))):
            assert result["lag_1h"].iloc[i] == pytest.approx(df["load_mw"].iloc[i - 1])

    def test_first_row_is_nan(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_lag_features(sample_clean_df.copy(), [1])
        assert pd.isna(result["lag_1h"].iloc[0])


class TestRollingFeatures:
    def test_creates_mean_and_std(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_rolling_features(sample_clean_df.copy(), [6])
        assert "rolling_mean_6h" in result.columns
        assert "rolling_std_6h" in result.columns

    def test_uses_shift_before_rolling(self, sample_clean_df: pd.DataFrame) -> None:
        """Rolling features must not include the current observation (look-ahead)."""
        df = sample_clean_df.copy()
        result = create_rolling_features(df, [2])
        # At position 2, rolling_mean_2h uses positions 0,1 (shifted by 1 then window=2)
        # It must NOT include position 2 itself
        expected = df["load_mw"].iloc[0:2].mean()
        actual = result["rolling_mean_2h"].iloc[2]
        assert actual == pytest.approx(expected, rel=1e-5)


class TestCalendarFeatures:
    def test_expected_columns(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_clean_df.copy(), "DE")
        for col in ["hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_working_day"]:
            assert col in result.columns

    def test_hour_range(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_clean_df.copy(), "DE")
        assert result["hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_clean_df.copy(), "DE")
        assert result["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_calendar_features(sample_clean_df.copy(), "DE")
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestFourierFeatures:
    def test_correct_column_count(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_fourier_features(sample_clean_df.copy(), [24, 168], order=2)
        # 2 periods × 2 orders × 2 (sin/cos) = 8 new columns
        new_cols = [c for c in result.columns if c.startswith("fourier_")]
        assert len(new_cols) == 8

    def test_values_bounded(self, sample_clean_df: pd.DataFrame) -> None:
        result = create_fourier_features(sample_clean_df.copy(), [24], order=1)
        sin_col = "fourier_24h_sin_1"
        cos_col = "fourier_24h_cos_1"
        assert result[sin_col].between(-1, 1).all()
        assert result[cos_col].between(-1, 1).all()
