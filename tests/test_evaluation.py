"""
Tests for src/evaluation.py — RMSE, MAE, MAPE, R² implementations.
"""
import numpy as np
import pytest
from src.evaluation import rmse, mae, mape, r_squared, compute_all_metrics


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # each error = 1, MSE = 1, RMSE = 1
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_non_negative(self):
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert rmse(y_true, y_pred) >= 0

    def test_symmetric(self):
        y_true = np.array([1.0, 4.0])
        y_pred = np.array([4.0, 1.0])
        assert rmse(y_true, y_pred) == rmse(y_pred, y_true)

    def test_single_element(self):
        assert rmse(np.array([5.0]), np.array([3.0])) == pytest.approx(2.0)

    def test_gold_target_scale(self):
        """RMSE on gold should be in dollar range (~10-50)."""
        y_true = np.random.uniform(1700, 1900, 200)
        y_pred = y_true + np.random.normal(0, 20, 200)
        result = rmse(y_true, y_pred)
        assert 0 < result < 200


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 0.0, 3.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_leq_rmse(self):
        """MAE ≤ RMSE always."""
        rng = np.random.RandomState(0)
        y_true = rng.randn(200)
        y_pred = rng.randn(200)
        assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-9


class TestMAPE:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # 10% error on every element
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_skips_zero_actual(self):
        """MAPE should not divide by zero; skips zero-valued actuals."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([1.0, 110.0])
        result = mape(y_true, y_pred)
        assert np.isfinite(result)
        assert result == pytest.approx(10.0)

    def test_non_negative(self):
        assert mape(np.array([1.0, 2.0]), np.array([1.5, 2.5])) >= 0


class TestR2:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self):
        """Predicting the mean everywhere → R² = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full_like(y_true, np.mean(y_true))
        assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_can_be_negative(self):
        """Terrible model that predicts opposite direction: R² < 0."""
        # y_true increases, y_pred decreases → ss_res > ss_tot → R² < 0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        assert r_squared(y_true, y_pred) < 0

    def test_constant_actual(self):
        """If actual is constant, R² = 1 only when pred also constant."""
        y_true = np.full(5, 3.0)
        assert r_squared(y_true, y_true) == pytest.approx(1.0)

    def test_high_r2_for_good_model(self):
        y_true = np.linspace(100, 200, 500)
        noise = np.random.default_rng(0).normal(0, 1, 500)
        y_pred = y_true + noise
        assert r_squared(y_true, y_pred) > 0.99


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        m = compute_all_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert set(m.keys()) == {"RMSE", "MAE", "MAPE", "R2"}

    def test_values_are_finite(self):
        rng = np.random.RandomState(7)
        y_true = rng.uniform(1, 100, 50)
        y_pred = y_true + rng.normal(0, 3, 50)
        m = compute_all_metrics(y_true, y_pred)
        for v in m.values():
            assert np.isfinite(v), f"Non-finite metric value: {v}"
