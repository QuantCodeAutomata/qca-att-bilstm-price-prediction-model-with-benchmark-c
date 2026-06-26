"""
Experiment 4: Full Trading Strategy Simulation with Modified Greedy Algorithm
and VaR Risk Management.

Simulates complete trading strategy for gold and bitcoin using Att-BiLSTM
signals, exponential position sizing, VaR filtering, and a modified greedy
decision algorithm. Includes transaction fee sensitivity analysis.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trading_strategy import simulate_trading, run_sensitivity_analysis

RESULTS_DIR = "results"
PRO = 500.0
INITIAL_CAPITAL = 1000.0

GOLD_BASE_FEE = 0.002   # 0.2%
BTC_BASE_FEE = 0.015    # 1.5%
GOLD_FEES = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
BTC_FEES = [0.015, 0.020, 0.025, 0.030, 0.035]


def _plot_portfolio(
    portfolio_history: np.ndarray,
    asset_name: str,
    suffix: str,
    initial: float = 500.0,
) -> str:
    """Plot portfolio value over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(portfolio_history)), portfolio_history, color="steelblue", lw=1.2)
    ax.axhline(initial, color="red", lw=0.8, ls="--", label=f"Initial ${initial:.0f}")
    ax.set_title(f"{asset_name} — Portfolio Value Over Time")
    ax.set_xlabel("Test Day Index")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    final_val = portfolio_history[-1]
    ret = (final_val - initial) / initial * 100
    ax.annotate(
        f"Final: ${final_val:,.0f} ({ret:+.1f}%)",
        xy=(len(portfolio_history) - 1, final_val),
        xytext=(-100, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, f"exp4_portfolio_{suffix}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def _plot_sensitivity(
    sensitivity: dict,
    asset_name: str,
    suffix: str,
    initial: float = 500.0,
) -> str:
    """Plot final portfolio value vs transaction fee rate."""
    fees = sorted(sensitivity.keys())
    finals = [sensitivity[f] for f in fees]
    fees_pct = [f * 100 for f in fees]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fees_pct, finals, "o-", color="darkblue", lw=1.5)
    ax.axhline(initial, color="red", lw=0.8, ls="--", label=f"Initial ${initial:.0f}")
    ax.set_title(f"{asset_name} — Final Portfolio Value vs Transaction Fee")
    ax.set_xlabel("Transaction Fee (%)")
    ax.set_ylabel("Final Portfolio Value ($)")
    ax.legend()
    for x, y in zip(fees_pct, finals):
        ax.annotate(f"${y:,.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, f"exp4_sensitivity_{suffix}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def run_exp4(exp1_results: dict = None, exp3_results: dict = None) -> dict:
    """Run Experiment 4: complete trading strategy simulation.

    Args:
        exp1_results: Output dict from run_exp1(). Must contain:
            gold/btc keys with 'test_prices', 'M_series', 'forecasts_3day_raw'.
        exp3_results: Output dict from run_exp3(). Must contain:
            gold/btc keys with 'y_values' and 'params' dict with u_n values.

    Returns:
        Dictionary with simulation results for 'gold' and 'btc'.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Resolve inputs (use provided or load fallback)
    # ------------------------------------------------------------------ #
    if exp1_results is not None:
        gold_test_prices = exp1_results["gold"]["test_prices"]
        btc_test_prices = exp1_results["btc"]["test_prices"]
        M_gold = exp1_results["gold"]["M_series"]
        M_btc = exp1_results["btc"]["M_series"]
        gold_forecasts = exp1_results["gold"]["forecasts_3day_raw"]
        btc_forecasts = exp1_results["btc"]["forecasts_3day_raw"]
    else:
        from src.data_loader import get_prices, compute_M_series, normalize_prices, create_sliding_windows, train_test_split_temporal
        from src.models import build_att_bilstm, train_dl_model, generate_multistep_forecasts

        print("  Generating predictions from scratch (no exp1 results provided)...")
        WINDOW = 100

        for ticker, name in [("GC=F", "gold"), ("BTC-USD", "btc")]:
            prices = get_prices(ticker)
            M = compute_M_series(prices)
            prices_norm, scaler = normalize_prices(prices)
            X, y = create_sliding_windows(prices_norm, WINDOW)
            X_train, X_test, y_train, y_test, split_idx = train_test_split_temporal(X, y)

            model = build_att_bilstm(WINDOW)
            train_dl_model(model, X_train, y_train)
            raw_split = WINDOW + split_idx

            if name == "gold":
                gold_test_prices = pd.Series(
                    scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                    index=prices.index[raw_split : raw_split + len(y_test)],
                )
                M_gold = M
                gold_forecasts = scaler.inverse_transform(
                    generate_multistep_forecasts(model, prices_norm, raw_split, WINDOW, 3).reshape(-1, 1)
                ).reshape(-1, 3)
            else:
                btc_test_prices = pd.Series(
                    scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                    index=prices.index[raw_split : raw_split + len(y_test)],
                )
                M_btc = M
                btc_forecasts = scaler.inverse_transform(
                    generate_multistep_forecasts(model, prices_norm, raw_split, WINDOW, 3).reshape(-1, 1)
                ).reshape(-1, 3)

    # ------------------------------------------------------------------ #
    # Resolve position sizing (from exp3 or defaults)
    # ------------------------------------------------------------------ #
    if exp3_results is not None:
        y_gold = exp3_results["gold"]["y_values"]
        y_btc = exp3_results["btc"]["y_values"]
        gold_u_n = exp3_results["params"]["gold_u_n"]
        btc_u_n = exp3_results["params"]["btc_u_n"]
    else:
        print("  Using default position sizing (run exp_3 for actual values).")
        from src.position_sizing import full_position_sizing
        gold_ps = full_position_sizing(4, 0.0042, -0.0080, PRO=500.0)
        btc_ps = full_position_sizing(5, 0.0150, -0.0350, PRO=500.0)
        y_gold = gold_ps["y_values"]
        y_btc = btc_ps["y_values"]
        gold_u_n = 4
        btc_u_n = 5

    print(f"  Gold  u_n={gold_u_n}, allocations: {[f'${v:.2f}' for v in y_gold]}")
    print(f"  Bitcoin u_n={btc_u_n}, allocations: {[f'${v:.2f}' for v in y_btc]}")

    # ------------------------------------------------------------------ #
    # Base simulation
    # ------------------------------------------------------------------ #
    print("\n  Running gold base simulation...")
    gold_hist, gold_trades = simulate_trading(
        gold_test_prices, M_gold, gold_forecasts,
        y_gold, gold_u_n, PRO=PRO, fee_rate=GOLD_BASE_FEE,
    )
    gold_final = float(gold_hist[-1])
    gold_return = (gold_final - PRO) / PRO * 100
    print(f"  Gold final: ${gold_final:,.2f} ({gold_return:+.2f}%)")
    print(f"  Gold trades: {len(gold_trades)}")

    print("  Running Bitcoin base simulation...")
    btc_hist, btc_trades = simulate_trading(
        btc_test_prices, M_btc, btc_forecasts,
        y_btc, btc_u_n, PRO=PRO, fee_rate=BTC_BASE_FEE,
    )
    btc_final = float(btc_hist[-1])
    btc_return = (btc_final - PRO) / PRO * 100
    print(f"  Bitcoin final: ${btc_final:,.2f} ({btc_return:+.2f}%)")
    print(f"  Bitcoin trades: {len(btc_trades)}")

    total_final = gold_final + btc_final
    total_return = (total_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"\n  Combined: ${total_final:,.2f} ({total_return:+.2f}%)")

    # ------------------------------------------------------------------ #
    # Sensitivity analysis
    # ------------------------------------------------------------------ #
    print("\n  Running gold fee sensitivity analysis...")
    gold_sensitivity = run_sensitivity_analysis(
        gold_test_prices, M_gold, gold_forecasts,
        y_gold, gold_u_n, PRO=PRO, fee_range=GOLD_FEES,
    )
    print("  Gold sensitivity:", {f"{k*100:.1f}%": f"${v:,.0f}" for k, v in gold_sensitivity.items()})

    print("  Running Bitcoin fee sensitivity analysis...")
    btc_sensitivity = run_sensitivity_analysis(
        btc_test_prices, M_btc, btc_forecasts,
        y_btc, btc_u_n, PRO=PRO, fee_range=BTC_FEES,
    )
    print("  Bitcoin sensitivity:", {f"{k*100:.1f}%": f"${v:,.0f}" for k, v in btc_sensitivity.items()})

    # ------------------------------------------------------------------ #
    # Verify monotone decreasing sensitivity
    # ------------------------------------------------------------------ #
    gold_vals = [gold_sensitivity[f] for f in sorted(gold_sensitivity)]
    btc_vals = [btc_sensitivity[f] for f in sorted(btc_sensitivity)]
    gold_mono = all(gold_vals[i] >= gold_vals[i + 1] for i in range(len(gold_vals) - 1))
    btc_mono = all(btc_vals[i] >= btc_vals[i + 1] for i in range(len(btc_vals) - 1))
    print(f"  Gold monotone decreasing: {gold_mono}")
    print(f"  Bitcoin monotone decreasing: {btc_mono}")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    summary = {
        "Asset": ["Gold", "Bitcoin", "Combined"],
        "Initial ($)": [PRO, PRO, INITIAL_CAPITAL],
        "Final ($)": [gold_final, btc_final, total_final],
        "Return (%)": [gold_return, btc_return, total_return],
        "Trades": [len(gold_trades), len(btc_trades), len(gold_trades) + len(btc_trades)],
    }
    pd.DataFrame(summary).to_csv(
        os.path.join(RESULTS_DIR, "exp4_simulation_summary.csv"), index=False
    )

    # Sensitivity tables
    gold_sens_df = pd.DataFrame([
        {"Fee (%)": round(f * 100, 1), "Final Portfolio ($)": round(v, 2)}
        for f, v in gold_sensitivity.items()
    ])
    gold_sens_df.to_csv(
        os.path.join(RESULTS_DIR, "exp4_gold_sensitivity.csv"), index=False
    )
    btc_sens_df = pd.DataFrame([
        {"Fee (%)": round(f * 100, 1), "Final Portfolio ($)": round(v, 2)}
        for f, v in btc_sensitivity.items()
    ])
    btc_sens_df.to_csv(
        os.path.join(RESULTS_DIR, "exp4_btc_sensitivity.csv"), index=False
    )

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("  Generating plots...")
    _plot_portfolio(gold_hist, "Gold", "gold", initial=PRO)
    _plot_portfolio(btc_hist, "Bitcoin", "btc", initial=PRO)
    _plot_sensitivity(gold_sensitivity, "Gold", "gold", initial=PRO)
    _plot_sensitivity(btc_sensitivity, "Bitcoin", "btc", initial=PRO)

    print(f"\n  Summary:")
    print(f"    Gold:    ${PRO:.0f} → ${gold_final:,.2f} ({gold_return:+.2f}%)")
    print(f"    Bitcoin: ${PRO:.0f} → ${btc_final:,.2f} ({btc_return:+.2f}%)")
    print(f"    Total:   ${INITIAL_CAPITAL:.0f} → ${total_final:,.2f} ({total_return:+.2f}%)")
    print("  Experiment 4 complete. Results saved to results/exp4_*")

    return {
        "gold": {
            "history": gold_hist,
            "final": gold_final,
            "return_pct": gold_return,
            "trades": gold_trades,
            "sensitivity": gold_sensitivity,
        },
        "btc": {
            "history": btc_hist,
            "final": btc_final,
            "return_pct": btc_return,
            "trades": btc_trades,
            "sensitivity": btc_sensitivity,
        },
        "total_final": total_final,
        "total_return_pct": total_return,
    }


if __name__ == "__main__":
    run_exp4()
