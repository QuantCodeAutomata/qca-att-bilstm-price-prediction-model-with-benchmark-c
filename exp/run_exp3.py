"""
Experiment 3: Exponential Position Sizing Model Derivation.

Derives {p_i} from break-even conditions (paper Eq. 6), fits
s_i = a·exp(b·i), and computes dollar allocations y_i for gold and Bitcoin.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.position_sizing import full_position_sizing

RESULTS_DIR = "results"
PRO = 500.0


def _plot_allocations(
    y_gold: np.ndarray,
    y_btc: np.ndarray,
) -> str:
    """Plot y_i allocations for both assets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y, label, color in [
        (axes[0], y_gold, "Gold", "goldenrod"),
        (axes[1], y_btc, "Bitcoin", "steelblue"),
    ]:
        i_vals = np.arange(1, len(y) + 1)
        ax.bar(i_vals, y, color=color, edgecolor="black")
        ax.set_title(f"{label} — Dollar Allocation y_i")
        ax.set_xlabel("Position Step i")
        ax.set_ylabel("Dollar Amount ($)")
        ax.set_xticks(i_vals)
        for xi, yi in zip(i_vals, y):
            ax.text(xi, yi + max(y) * 0.01, f"${yi:.1f}", ha="center", fontsize=8)

    plt.suptitle("Exponential Position Sizing Allocations (PRO=$500 per asset)")
    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, "exp3_allocations.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def _plot_exponential_fit(
    positions_gold: np.ndarray,
    s_gold: np.ndarray,
    positions_btc: np.ndarray,
    s_btc: np.ndarray,
) -> str:
    """Plot derived p_i and fitted s_i on log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, positions, s, label in [
        (axes[0], positions_gold, s_gold, "Gold"),
        (axes[1], positions_btc, s_btc, "Bitcoin"),
    ]:
        i_vals = np.arange(1, len(positions) + 1)
        ax.semilogy(i_vals, positions, "o", label="Derived p_i", color="black")
        ax.semilogy(i_vals, s, "-", label="Fitted s_i = a·exp(b·i)", color="red")
        ax.set_title(f"{label} — Exponential Fit")
        ax.set_xlabel("Position Step i")
        ax.set_ylabel("Position Size (log scale)")
        ax.legend()
        ax.set_xticks(i_vals)

    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, "exp3_exponential_fit.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def run_exp3(params: dict = None) -> dict:
    """Run Experiment 3: Exponential position sizing.

    Args:
        params: Optional pre-computed parameters from exp_2. Dict with keys:
            'gold_u_n', 'gold_M_u05', 'gold_M_d01',
            'btc_u_n', 'btc_M_u05', 'btc_M_d01'.
            Falls back to sensible defaults if not provided.

    Returns:
        Dictionary with sizing results for 'gold' and 'btc'.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Use exp_2 outputs or sensible defaults
    if params is None:
        print("  Using default parameters (run exp_2 first for actual values).")
        params = {
            "gold_u_n": 4,
            "gold_M_u05": 0.0042,
            "gold_M_d01": -0.0080,
            "btc_u_n": 5,
            "btc_M_u05": 0.0150,
            "btc_M_d01": -0.0350,
        }

    print(f"  Gold  : u_n={params['gold_u_n']}, M_u05={params['gold_M_u05']:.4f}, "
          f"M_d01={params['gold_M_d01']:.4f}")
    print(f"  Bitcoin: u_n={params['btc_u_n']}, M_u05={params['btc_M_u05']:.4f}, "
          f"M_d01={params['btc_M_d01']:.4f}")

    # ------- Gold -------
    gold_sizing = full_position_sizing(
        u_n=params["gold_u_n"],
        M_u05=params["gold_M_u05"],
        M_d01=params["gold_M_d01"],
        PRO=PRO,
    )
    print(f"\n  Gold exponential fit: a={gold_sizing['a']:.4f}, "
          f"b={gold_sizing['b']:.4f}, R²={gold_sizing['r_squared']:.4f}")
    print(f"  Gold allocations (sum=${np.sum(gold_sizing['y_values']):.2f}): "
          f"{[f'${v:.2f}' for v in gold_sizing['y_values']]}")

    # ------- Bitcoin -------
    btc_sizing = full_position_sizing(
        u_n=params["btc_u_n"],
        M_u05=params["btc_M_u05"],
        M_d01=params["btc_M_d01"],
        PRO=PRO,
    )
    print(f"\n  Bitcoin exponential fit: a={btc_sizing['a']:.4f}, "
          f"b={btc_sizing['b']:.4f}, R²={btc_sizing['r_squared']:.4f}")
    print(f"  Bitcoin allocations (sum=${np.sum(btc_sizing['y_values']):.2f}): "
          f"{[f'${v:.2f}' for v in btc_sizing['y_values']]}")

    # ------- Validation -------
    for name, sizing in [("Gold", gold_sizing), ("Bitcoin", btc_sizing)]:
        assert abs(np.sum(sizing["y_values"]) - PRO) < 1e-4, \
            f"{name} allocations don't sum to {PRO}"
        assert np.all(np.diff(sizing["y_values"]) > 0), \
            f"{name} allocations are not monotonically increasing"
        assert sizing["r_squared"] > 0.95, \
            f"{name} exponential fit R² = {sizing['r_squared']:.4f} < 0.95"
        print(f"  {name} validation: PASSED (R²={sizing['r_squared']:.4f}, "
              f"monotone={np.all(np.diff(sizing['y_values'])>0)})")

    # ------- Summary table -------
    rows = []
    for name, sizing, u_n_key in [
        ("Gold", gold_sizing, "gold_u_n"),
        ("Bitcoin", btc_sizing, "btc_u_n"),
    ]:
        u_n = params[u_n_key]
        for i in range(u_n):
            rows.append({
                "Asset": name,
                "i": i + 1,
                "p_i (derived)": round(sizing["positions"][i], 6),
                "s_i (fitted)": round(sizing["s_values"][i], 6),
                "y_i ($)": round(sizing["y_values"][i], 4),
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "exp3_position_sizing.csv"), index=False)
    print("\n  Position sizing table:")
    print(summary_df.to_string(index=False))

    # ------- Plots -------
    print("  Generating plots...")
    _plot_allocations(gold_sizing["y_values"], btc_sizing["y_values"])
    _plot_exponential_fit(
        gold_sizing["positions"], gold_sizing["s_values"],
        btc_sizing["positions"], btc_sizing["s_values"],
    )

    print("  Experiment 3 complete. Results saved to results/exp3_*")

    return {"gold": gold_sizing, "btc": btc_sizing, "params": params}


if __name__ == "__main__":
    run_exp3()
