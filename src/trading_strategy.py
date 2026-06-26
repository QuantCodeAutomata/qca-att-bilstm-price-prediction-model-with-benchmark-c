"""
Full trading strategy simulation with modified greedy algorithm and VaR filter.

Implements the complete pipeline:
  Att-BiLSTM signals → VaR filter → modified greedy decision
  → exponential position execution → portfolio tracking.

# Custom — Context7 found no library equivalent (paper Section 6)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_var(
    M_series: np.ndarray,
    t: int,
    lookback: int = 100,
    conf: float = 0.95,
) -> float:
    """Compute 1-day rolling historical VaR at time index t.

    Uses the past `lookback` M_i values ending at (but not including) t.
    VaR_95 = 5th percentile of the M window (i.e., worst 5% daily loss).

    Args:
        M_series: Full M_i array (all history, train + test).
        t: Current absolute index into M_series.
        lookback: Rolling window length (default 100).
        conf: Confidence level (default 0.95).

    Returns:
        VaR scalar (negative value representing loss threshold).
    """
    if t < lookback:
        return -0.05  # default 5% loss threshold when history insufficient
    window = M_series[t - lookback : t]
    return float(np.percentile(window, (1 - conf) * 100))


def generate_signals(
    prices: np.ndarray,
    predictions_3day: np.ndarray,
) -> np.ndarray:
    """Generate buy/sell/hold signals from 3-day ahead price forecasts.

    Signal logic uses prediction momentum (day-over-day change in the
    1-step-ahead forecast) to avoid the level-bias problem that arises
    when test prices exceed the training normalisation range.

    r_hat = (fc[0][t] - fc[0][t-1]) / |fc[0][t-1]|   [momentum]
    up_days = number of the last 3 days where the 1-step prediction rose

    BUY  (+1): r_hat > 0 AND up_days >= 2
    SELL (-1): r_hat < 0 OR  up_days <= 1
    HOLD  (0): otherwise

    Args:
        prices: Array of test-period actual prices (shape: n_test).
        predictions_3day: 3-day ahead forecasts in actual price scale,
            shape (n_test, 3). Row t = [p_{t+1}, p_{t+2}, p_{t+3}].

    Returns:
        Integer array of signals: +1=BUY, -1=SELL, 0=HOLD (shape: n_test).
    """
    n = len(predictions_3day)
    signals = np.zeros(len(prices), dtype=int)

    # Extract the 1-step-ahead forecast series
    first_step = predictions_3day[:, 0]  # shape (n,)

    for t in range(len(prices)):
        p_t = prices[t]

        if t >= n:
            signals[t] = -1
            continue

        fc = predictions_3day[t]

        # --- r_hat: momentum of the 1-step prediction ---
        if t > 0:
            fc_prev0 = first_step[t - 1]
        else:
            fc_prev0 = first_step[0]

        r_hat = (first_step[t] - fc_prev0) / (abs(fc_prev0) + 1e-10)

        # --- up_days: rising 1-step predictions over the past 3 days ---
        start = max(0, t - 2)
        window_preds = first_step[start : t + 1]
        up_days = int(sum(
            window_preds[k] > window_preds[k - 1]
            for k in range(1, len(window_preds))
        ))
        # Normalise to a [0, 3] scale if window is shorter than 3 days
        if len(window_preds) < 3:
            up_days = up_days + (3 - len(window_preds))  # assume neutral for missing days

        if r_hat > 0 and up_days >= 2:
            signals[t] = 1
        elif r_hat < 0 or up_days <= 1:
            signals[t] = -1
        else:
            signals[t] = 0

    return signals


def simulate_trading(
    prices: pd.Series,
    M_series: pd.Series,
    predictions_3day: np.ndarray,
    y_allocations: np.ndarray,
    u_n: int,
    PRO: float = 500.0,
    fee_rate: float = 0.002,
    var_lookback: int = 100,
) -> Tuple[np.ndarray, List[Dict]]:
    """Full trading simulation using modified greedy algorithm with VaR filter.

    State machine:
      - cash: available dollars (starts at PRO)
      - shares: units held
      - consec_fall: consecutive fall counter (resets on BUY or after u_n falls)
      - in_position: whether any shares are held

    On BUY signal:
      1. Check VaR: skip if expected_loss > 10% of portfolio value.
      2. Invest y_allocations[consec_fall] dollars.
      3. Units = net_investment / price;  net = invest * (1 - fee_rate).
      4. Reset consec_fall = 0.

    On SELL signal (in position):
      1. Liquidate all shares.
      2. net_proceeds = shares * price * (1 - fee_rate).
      3. consec_fall += 1; reset to 0 if >= u_n.

    Args:
        prices: Test-period price series (pd.Series with DatetimeIndex).
        M_series: Full M_i series (all history) used for VaR computation.
        predictions_3day: Shape (n_test, 3) actual-price forecasts.
        y_allocations: Dollar allocations per position step, shape (u_n,).
        u_n: Maximum consecutive fall count.
        PRO: Initial capital (default 500).
        fee_rate: Transaction fee rate (default 0.002 = 0.2%).
        var_lookback: Rolling window for VaR (default 100).

    Returns:
        Tuple of (portfolio_history: np.ndarray shape (n_test,),
                  trade_log: list of trade-event dicts).
    """
    prices_arr = np.asarray(prices.values, dtype=float)
    M_arr = np.asarray(M_series.values, dtype=float)
    n_test = len(prices_arr)
    # Absolute start index of test period in full M_series
    test_start_idx = len(M_arr) - n_test

    cash = PRO
    shares = 0.0
    consec_fall = 0
    in_position = False
    portfolio_history = np.empty(n_test, dtype=float)
    trade_log: List[Dict] = []

    signals = generate_signals(prices_arr, predictions_3day)

    for t in range(n_test):
        p_t = prices_arr[t]
        full_idx = test_start_idx + t

        # Rolling VaR from full history
        var_thresh = compute_var(M_arr, full_idx, lookback=var_lookback)

        sig = int(signals[t])

        if sig == 1:  # BUY signal
            pos_step = min(consec_fall, u_n - 1)
            invest = y_allocations[pos_step]

            if invest <= cash:
                # VaR check: expected loss on new position
                position_value = shares * p_t + invest
                expected_loss = abs(var_thresh) * position_value
                risk_limit = 0.10 * (cash + shares * p_t)

                if expected_loss <= risk_limit or risk_limit == 0:
                    units = (invest * (1.0 - fee_rate)) / p_t
                    shares += units
                    cash -= invest
                    consec_fall = 0
                    in_position = True
                    trade_log.append(
                        {
                            "day": t,
                            "action": "BUY",
                            "price": p_t,
                            "amount": invest,
                            "units": units,
                        }
                    )

        elif sig == -1:  # SELL signal
            if in_position and shares > 0:
                gross = shares * p_t
                net_proceeds = gross * (1.0 - fee_rate)
                cash += net_proceeds
                trade_log.append(
                    {
                        "day": t,
                        "action": "SELL",
                        "price": p_t,
                        "proceeds": net_proceeds,
                        "shares": shares,
                    }
                )
                shares = 0.0
                in_position = False
                consec_fall += 1
                if consec_fall >= u_n:
                    consec_fall = 0
            else:
                # Not in position: track actual fall
                M_t = M_arr[full_idx] if full_idx < len(M_arr) else 0.0
                if M_t < 0:
                    consec_fall = min(consec_fall + 1, u_n)
                elif M_t > 0:
                    consec_fall = 0

        # HOLD: no action; track M_t if not in position
        else:
            if not in_position:
                M_t = M_arr[full_idx] if full_idx < len(M_arr) else 0.0
                if M_t < 0:
                    consec_fall = min(consec_fall + 1, u_n)
                elif M_t > 0:
                    consec_fall = 0

        portfolio_history[t] = cash + shares * p_t

    return portfolio_history, trade_log


def run_sensitivity_analysis(
    prices: pd.Series,
    M_series: pd.Series,
    predictions_3day: np.ndarray,
    y_allocations: np.ndarray,
    u_n: int,
    PRO: float,
    fee_range: List[float],
) -> Dict[float, float]:
    """Run the full simulation for each fee rate and record final portfolio value.

    Args:
        prices: Test-period price series.
        M_series: Full M_i series.
        predictions_3day: 3-day forecasts.
        y_allocations: Dollar allocations per position step.
        u_n: Maximum consecutive fall count.
        PRO: Initial capital.
        fee_range: List of fee rates to test.

    Returns:
        Dictionary mapping fee_rate → final_portfolio_value.
    """
    results: Dict[float, float] = {}
    for fee in fee_range:
        hist, _ = simulate_trading(
            prices,
            M_series,
            predictions_3day,
            y_allocations,
            u_n,
            PRO=PRO,
            fee_rate=fee,
        )
        results[fee] = float(hist[-1])
    return results
