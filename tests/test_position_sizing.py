"""
Tests for src/position_sizing.py — break-even position derivation,
exponential fit, and allocation normalisation.
"""
import numpy as np
import pytest

from src.position_sizing import (
    derive_position_sizes,
    fit_exponential,
    compute_allocations,
    full_position_sizing,
)

# Gold defaults from paper calibration
GOLD_PARAMS = dict(u_n=4, M_u05=0.0042, M_d01=-0.0080)
BTC_PARAMS = dict(u_n=5, M_u05=0.0150, M_d01=-0.0350)
# High-volatility params where |M_d01/u_n| > M_u05 → positions are increasing
HIGH_VOL_PARAMS = dict(u_n=2, M_u05=0.003, M_d01=-0.010)


class TestDerivePositionSizes:
    def test_gold_u_n_4(self):
        positions = derive_position_sizes(**GOLD_PARAMS)
        assert len(positions) == 4

    def test_all_positive(self):
        positions = derive_position_sizes(**GOLD_PARAMS)
        assert np.all(positions > 0)

    def test_p1_equals_one(self):
        positions = derive_position_sizes(**GOLD_PARAMS)
        assert positions[0] == pytest.approx(1.0)

    def test_monotone_increasing(self):
        # Monotone increasing requires |M_d01/u_n| > M_u05 (high-volatility regime)
        positions = derive_position_sizes(**HIGH_VOL_PARAMS)
        assert np.all(np.diff(positions) > 0)

    def test_btc_u_n_5(self):
        positions = derive_position_sizes(**BTC_PARAMS)
        assert len(positions) == 5
        assert np.all(positions > 0)

    def test_u_n_1_returns_base(self):
        positions = derive_position_sizes(1, 0.005, -0.01)
        assert len(positions) == 1
        assert positions[0] == pytest.approx(1.0)

    def test_u_n_2_known(self):
        """For u_n=2: p_2 = -p_1 * (M_d01/u_n) / M_u05."""
        M_u05, M_d01, u_n = 0.004, -0.008, 2
        positions = derive_position_sizes(u_n, M_u05, M_d01)
        expected_p2 = -1.0 * (M_d01 / u_n) / M_u05  # = 0.004 / 0.004 = 1.0
        assert positions[1] == pytest.approx(expected_p2, rel=1e-6)

    def test_larger_u_n_produces_more_positions(self):
        pos4 = derive_position_sizes(4, 0.005, -0.008)
        pos6 = derive_position_sizes(6, 0.005, -0.008)
        assert len(pos4) == 4
        assert len(pos6) == 6

    def test_invalid_M_d01_sign_raises(self):
        """Positive M_d01 should produce non-positive positions → ValueError."""
        with pytest.raises((ValueError, Exception)):
            derive_position_sizes(3, 0.004, 0.008)


class TestFitExponential:
    def test_r_squared_gt_095(self):
        positions = derive_position_sizes(**GOLD_PARAMS)
        _, _, r2 = fit_exponential(positions)
        assert r2 > 0.95

    def test_a_positive(self):
        positions = derive_position_sizes(**GOLD_PARAMS)
        a, _, _ = fit_exponential(positions)
        assert a > 0

    def test_b_positive_for_growing_positions(self):
        # b > 0 only when positions are increasing (high-volatility regime)
        positions = derive_position_sizes(**HIGH_VOL_PARAMS)
        _, b, _ = fit_exponential(positions)
        assert b > 0

    def test_perfect_exponential(self):
        """Perfectly exponential data → R² = 1."""
        positions = np.exp(np.arange(1, 6, dtype=float))
        a, b, r2 = fit_exponential(positions)
        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_fitted_values_match(self):
        positions = derive_position_sizes(**GOLD_PARAMS)
        a, b, _ = fit_exponential(positions)
        i_vals = np.arange(1, len(positions) + 1, dtype=float)
        s_fitted = a * np.exp(b * i_vals)
        # Fitted values should be close to derived
        np.testing.assert_allclose(s_fitted, positions, rtol=0.10)


class TestComputeAllocations:
    def test_sum_equals_PRO(self):
        s_vals = np.array([1.0, 2.0, 4.0, 8.0])
        y = compute_allocations(s_vals, PRO=500.0)
        assert abs(np.sum(y) - 500.0) < 1e-6

    def test_proportional(self):
        s_vals = np.array([1.0, 3.0])
        y = compute_allocations(s_vals, PRO=100.0)
        assert y[0] == pytest.approx(25.0)
        assert y[1] == pytest.approx(75.0)

    def test_single_position(self):
        y = compute_allocations(np.array([5.0]), PRO=500.0)
        assert y[0] == pytest.approx(500.0)


class TestFullPositionSizing:
    def test_gold_output_keys(self):
        result = full_position_sizing(**GOLD_PARAMS, PRO=500.0)
        required_keys = {"positions", "s_values", "y_values", "a", "b", "r_squared"}
        assert required_keys.issubset(set(result.keys()))

    def test_sum_equals_PRO(self):
        result = full_position_sizing(**GOLD_PARAMS, PRO=500.0)
        assert abs(np.sum(result["y_values"]) - 500.0) < 1e-4

    def test_monotone_y_values(self):
        # Monotone increasing y_i requires high-volatility parameters
        result = full_position_sizing(**HIGH_VOL_PARAMS, PRO=500.0)
        assert np.all(np.diff(result["y_values"]) > 0), "y_i must be monotonically increasing"

    def test_r_squared_quality(self):
        result = full_position_sizing(**GOLD_PARAMS, PRO=500.0)
        assert result["r_squared"] > 0.95

    def test_btc_full_pipeline(self):
        result = full_position_sizing(**BTC_PARAMS, PRO=500.0)
        assert len(result["y_values"]) == 5
        assert abs(np.sum(result["y_values"]) - 500.0) < 1e-4
        assert result["r_squared"] > 0.95

    def test_y_values_length_equals_u_n(self):
        for u_n in [2, 3, 4, 5]:
            result = full_position_sizing(u_n, 0.005, -0.010, PRO=500.0)
            assert len(result["y_values"]) == u_n
