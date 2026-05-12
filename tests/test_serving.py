"""Tests for the serving-side calibration utilities.

These functions ship inside the registered MLflow model, so they need to
behave the same way at training time and at inference time. The tests
exercise both the static split-conformal offset and the streaming ACI
update on small synthetic sequences.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlops_forecast.serving.quantile_lgbm import (
    adaptive_conformal_offsets,
    conformal_offset,
)


@pytest.fixture
def calibration_set() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Targets and raw quantile predictions where the raw 80% interval covers ~60%."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(0, 10, n)
    # Symmetric raw bounds at +/- 5 cover only the inner ~38% of a N(0, 10);
    # so the conformal offset needs to widen them substantially.
    y_lower = -5 * np.ones(n)
    y_upper = 5 * np.ones(n)
    return y, y_lower, y_upper


class TestSplitConformal:
    def test_offset_is_nonnegative_when_raw_undercovers(
        self, calibration_set: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        y, lo, hi = calibration_set
        q = conformal_offset(y, lo, hi, alpha=0.2)
        assert q > 0  # raw interval undercovers, so we have to widen

    def test_calibrated_interval_hits_nominal_coverage(
        self, calibration_set: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """On the calibration set, the post-hoc widened interval should
        cover at least the nominal level. Split-conformal guarantees this."""
        y, lo, hi = calibration_set
        q = conformal_offset(y, lo, hi, alpha=0.2)
        cov = ((y >= lo - q) & (y <= hi + q)).mean()
        assert cov >= 0.80

    def test_offset_can_be_negative_when_raw_overcovers(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 300)
        lo = -10 * np.ones(300)
        hi = 10 * np.ones(300)
        q = conformal_offset(y, lo, hi, alpha=0.2)
        assert q < 0  # raw interval is way too wide, conformal narrows it


class TestACI:
    def test_aci_returns_arrays_of_correct_length(
        self, calibration_set: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        y, lo, hi = calibration_set
        q_hist, hits = adaptive_conformal_offsets(y, lo, hi, target_coverage=0.8)
        assert q_hist.shape == y.shape
        assert hits.shape == y.shape
        assert hits.dtype.kind in {"i", "u"}

    def test_aci_drives_coverage_toward_target(self) -> None:
        """ACI should pull empirical coverage toward the target level on a
        sequence that starts undercovered."""
        rng = np.random.default_rng(2)
        n = 2_000
        y = rng.normal(0, 10, n)
        # Heavy undercoverage to start with.
        lo = -2 * np.ones(n)
        hi = 2 * np.ones(n)
        _, hits = adaptive_conformal_offsets(y, lo, hi, target_coverage=0.8, gamma=0.05)
        # On the back half the running ACI offset has had time to converge;
        # we want it within +/- 5 percentage points of nominal.
        late = hits[n // 2 :].mean()
        assert 0.75 <= late <= 0.85
