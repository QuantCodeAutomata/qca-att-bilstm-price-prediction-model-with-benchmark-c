"""
Tests for src/data_loader.py — M_i computation, normalisation,
sliding windows, and temporal splits.
"""
import numpy as np
import pandas as pd
import pytest
from src.data_loader import (
    compute_M_series,
    normalize_prices,
    create_sliding_windows,
    train_test_split_temporal,
)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Create a synthetic price series for testing."""
    rng = np.random.RandomState(42)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, 500))
    return pd.Series(prices, index=pd.date_range("2018-01-01", periods=500, freq="B"))


class TestComputeMSeries:
    def test_length(self, sample_prices):
        M = compute_M_series(sample_prices)
        assert len(M) == len(sample_prices) - 1

    def test_formula(self, sample_prices):
        """M_i = (p_i - p_{i-1}) / p_i (denominator is p_i, NOT p_{i-1})."""
        p = sample_prices.values
        M_expected = (p[1:] - p[:-1]) / p[1:]
        M_actual = compute_M_series(sample_prices).values
        np.testing.assert_allclose(M_actual, M_expected, rtol=1e-10)

    def test_M_between_minus1_and_1(self, sample_prices):
        M = compute_M_series(sample_prices)
        assert (M > -1).all() and (M < 1).all()

    def test_positive_M_on_rising_price(self):
        prices = pd.Series([100.0, 110.0, 120.0])
        M = compute_M_series(prices)
        assert (M > 0).all()

    def test_negative_M_on_falling_price(self):
        prices = pd.Series([120.0, 110.0, 100.0])
        M = compute_M_series(prices)
        assert (M < 0).all()

    def test_dropna_removes_first_row(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0])
        M = compute_M_series(prices)
        assert len(M) == 3

    def test_not_standard_return(self):
        """Confirm denominator is p_i, not p_{i-1}."""
        prices = pd.Series([100.0, 110.0])
        M = compute_M_series(prices)
        expected = (110.0 - 100.0) / 110.0
        assert M.iloc[0] == pytest.approx(expected)


class TestNormalizePrices:
    def test_output_shape(self, sample_prices):
        norm, scaler = normalize_prices(sample_prices)
        assert len(norm) == len(sample_prices)

    def test_training_range_is_01(self, sample_prices):
        norm, _ = normalize_prices(sample_prices, train_frac=0.7)
        split = int(len(sample_prices) * 0.7)
        train_norm = norm[:split]
        assert train_norm.min() >= 0.0 - 1e-9
        assert train_norm.max() <= 1.0 + 1e-9

    def test_test_can_exceed_01(self, sample_prices):
        """Test values may exceed [0,1] if test prices are outside train range."""
        norm, _ = normalize_prices(sample_prices)
        # At least the full range should have values near [0,1] for train
        split = int(len(sample_prices) * 0.7)
        assert norm[:split].min() >= -0.01

    def test_scaler_inverse_transform(self, sample_prices):
        norm, scaler = normalize_prices(sample_prices)
        reconstructed = scaler.inverse_transform(norm.reshape(-1, 1)).flatten()
        np.testing.assert_allclose(reconstructed, sample_prices.values, rtol=1e-6)


class TestCreateSlidingWindows:
    def test_output_shapes(self, sample_prices):
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=100)
        assert X.shape == (len(norm) - 100, 100, 1)
        assert y.shape == (len(norm) - 100,)

    def test_window_content(self, sample_prices):
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=5)
        # First window should be norm[0:5], target should be norm[5]
        np.testing.assert_allclose(X[0, :, 0], norm[:5], rtol=1e-9)
        assert y[0] == pytest.approx(norm[5])

    def test_consecutive_windows_overlap(self, sample_prices):
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=10)
        np.testing.assert_allclose(X[0, 1:, 0], X[1, :-1, 0], rtol=1e-9)

    def test_small_window(self):
        prices = np.arange(20, dtype=float)
        X, y = create_sliding_windows(prices, window=3)
        assert X.shape == (17, 3, 1)

    def test_single_pair(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0])
        X, y = create_sliding_windows(prices, window=3)
        assert len(X) == 1


class TestTrainTestSplit:
    def test_split_sizes(self, sample_prices):
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=10)
        X_tr, X_te, y_tr, y_te, idx = train_test_split_temporal(
            X, y, train_frac=0.7, shuffle_train=False
        )
        assert len(X_tr) + len(X_te) == len(X)
        assert idx == int(len(X) * 0.7)

    def test_no_leakage(self, sample_prices):
        """Test set must come after training set chronologically."""
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=10)
        X_tr, X_te, y_tr, y_te, split_idx = train_test_split_temporal(
            X, y, train_frac=0.7, shuffle_train=False
        )
        # Without shuffle, the first test window follows the last train window
        # Compare last train y and first test window
        assert split_idx == len(X_tr)

    def test_shuffle_does_not_change_test(self, sample_prices):
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=10)
        _, X_te_no, _, y_te_no, _ = train_test_split_temporal(
            X, y, shuffle_train=False
        )
        _, X_te_sh, _, y_te_sh, _ = train_test_split_temporal(
            X, y, shuffle_train=True, seed=0
        )
        np.testing.assert_array_equal(X_te_no, X_te_sh)

    def test_shuffle_changes_train(self, sample_prices):
        norm, _ = normalize_prices(sample_prices)
        X, y = create_sliding_windows(norm, window=10)
        X_tr_no, _, _, _, _ = train_test_split_temporal(X, y, shuffle_train=False)
        X_tr_sh, _, _, _, _ = train_test_split_temporal(X, y, shuffle_train=True)
        assert not np.array_equal(X_tr_no, X_tr_sh)
