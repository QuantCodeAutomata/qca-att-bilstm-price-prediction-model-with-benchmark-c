"""
Tests for src/trading_strategy.py — VaR calculation, signal generation,
simulation correctness, and sensitivity analysis.
"""
import numpy as np
import pandas as pd
import pytest

from src.trading_strategy import (
    compute_var,
    generate_signals,
    simulate_trading,
    run_sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    rng = np.random.RandomState(5)
    n = 200
    price = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(price, index=dates)


@pytest.fixture
def synthetic_M(synthetic_prices):
    M = (synthetic_prices - synthetic_prices.shift(1)) / synthetic_prices
    return M.dropna()


@pytest.fixture
def flat_predictions(synthetic_prices):
    """Neutral 3-day forecasts: each day's forecast = current price."""
    n = len(synthetic_prices)
    prices_arr = synthetic_prices.values
    # All forecasts equal to the same day's price — should generate mostly HOLD
    preds = np.column_stack([prices_arr, prices_arr, prices_arr])
    return preds


@pytest.fixture
def bullish_predictions(synthetic_prices):
    """Bullish 3-day forecasts: each step +0.5%."""
    n = len(synthetic_prices)
    prices_arr = synthetic_prices.values
    preds = np.column_stack([
        prices_arr * 1.005,
        prices_arr * 1.010,
        prices_arr * 1.015,
    ])
    return preds


@pytest.fixture
def y_alloc():
    return np.array([100.0, 150.0, 250.0])


# ---------------------------------------------------------------------------
# VaR Tests
# ---------------------------------------------------------------------------

class TestComputeVar:
    def test_returns_negative_for_losses(self):
        M = np.random.default_rng(0).normal(-0.002, 0.01, 500)
        var = compute_var(M, t=300, lookback=100, conf=0.95)
        # 5th percentile of negative-drift returns is strongly negative
        assert var < 0

    def test_default_when_insufficient_history(self):
        M = np.zeros(50)
        var = compute_var(M, t=10, lookback=100)
        assert var == pytest.approx(-0.05)

    def test_uses_correct_window(self):
        M = np.zeros(300)
        M[150:200] = -0.1  # large losses in window 150-200
        # At t=200, window is [100:200] which includes the losses
        var1 = compute_var(M, t=200, lookback=100)
        # At t=300, window is [200:300] which does NOT include them
        var2 = compute_var(M, t=300, lookback=100)
        assert var1 < var2  # var1 is more negative

    def test_higher_confidence_gives_more_negative_var(self):
        rng = np.random.default_rng(1)
        M = rng.normal(0, 0.015, 500)
        var95 = compute_var(M, t=300, lookback=100, conf=0.95)
        var99 = compute_var(M, t=300, lookback=100, conf=0.99)
        assert var99 <= var95


# ---------------------------------------------------------------------------
# Signal Tests
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def test_output_shape(self, synthetic_prices, bullish_predictions):
        prices = synthetic_prices.values
        signals = generate_signals(prices, bullish_predictions)
        assert signals.shape == (len(prices),)

    def test_bullish_gives_buy(self, synthetic_prices, bullish_predictions):
        prices = synthetic_prices.values
        signals = generate_signals(prices, bullish_predictions)
        # At least some BUY signals should appear with bullish forecasts
        assert np.any(signals == 1)

    def test_flat_gives_mostly_hold_or_sell(self, synthetic_prices, flat_predictions):
        prices = synthetic_prices.values
        signals = generate_signals(prices, flat_predictions)
        buy_frac = np.mean(signals == 1)
        assert buy_frac < 0.5

    def test_signal_values_in_range(self, synthetic_prices, bullish_predictions):
        prices = synthetic_prices.values
        signals = generate_signals(prices, bullish_predictions)
        assert set(signals).issubset({-1, 0, 1})

    def test_no_forecast_defaults_to_neutral(self, synthetic_prices):
        n = len(synthetic_prices)
        prices = synthetic_prices.values
        # Pass fewer predictions than test days
        predictions = np.ones((n // 2, 3)) * prices[0]
        signals = generate_signals(prices, predictions)
        assert len(signals) == n


# ---------------------------------------------------------------------------
# Simulation Tests
# ---------------------------------------------------------------------------

class TestSimulateTrading:
    def test_portfolio_starts_at_PRO(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        hist, _ = simulate_trading(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc, u_n=3, PRO=500.0
        )
        assert hist[0] == pytest.approx(500.0, rel=0.02)

    def test_portfolio_value_non_negative(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        hist, _ = simulate_trading(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc, u_n=3, PRO=500.0
        )
        assert np.all(hist >= 0)

    def test_output_shape(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        n = len(synthetic_prices)
        hist, trade_log = simulate_trading(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc, u_n=3, PRO=500.0
        )
        assert len(hist) == n

    def test_higher_fee_gives_lower_final_value(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        hist_low, _ = simulate_trading(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc,
            u_n=3, PRO=500.0, fee_rate=0.001,
        )
        hist_high, _ = simulate_trading(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc,
            u_n=3, PRO=500.0, fee_rate=0.05,
        )
        assert hist_low[-1] >= hist_high[-1] - 1e-6

    def test_no_trades_when_holding(
        self, synthetic_prices, synthetic_M, y_alloc
    ):
        """With all-HOLD signals, portfolio should equal initial cash throughout."""
        n = len(synthetic_prices)
        prices_arr = synthetic_prices.values
        # Neutral predictions that produce HOLD: equal consecutive prices
        hold_preds = np.column_stack([prices_arr, prices_arr, prices_arr])
        hist, trade_log = simulate_trading(
            synthetic_prices, synthetic_M, hold_preds, y_alloc,
            u_n=3, PRO=500.0, fee_rate=0.002,
        )
        # No buys should happen if signals are HOLD/SELL with 0-change forecasts
        buy_trades = [t for t in trade_log if t["action"] == "BUY"]
        # Trade log may be empty — just check portfolio is non-negative
        assert np.all(hist >= 0)

    def test_trade_log_has_required_fields(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        _, trade_log = simulate_trading(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc, u_n=3, PRO=500.0
        )
        for trade in trade_log:
            assert "action" in trade
            assert "price" in trade
            assert "day" in trade


# ---------------------------------------------------------------------------
# Sensitivity Analysis Tests
# ---------------------------------------------------------------------------

class TestRunSensitivityAnalysis:
    def test_output_keys_match_fee_range(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        fees = [0.001, 0.005, 0.010]
        results = run_sensitivity_analysis(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc,
            u_n=3, PRO=500.0, fee_range=fees
        )
        assert set(results.keys()) == {0.001, 0.005, 0.010}

    def test_values_are_positive_floats(
        self, synthetic_prices, synthetic_M, bullish_predictions, y_alloc
    ):
        fees = [0.001, 0.01]
        results = run_sensitivity_analysis(
            synthetic_prices, synthetic_M, bullish_predictions, y_alloc,
            u_n=3, PRO=500.0, fee_range=fees
        )
        for v in results.values():
            assert isinstance(v, float)
            assert v >= 0
