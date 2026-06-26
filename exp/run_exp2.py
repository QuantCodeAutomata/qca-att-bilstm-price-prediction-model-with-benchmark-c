"""
Experiment 2: Apriori-Based Consecutive Rise/Fall Pattern Analysis
for Gold and Bitcoin.

Computes run-length statistics and key parameters (u_n, M_u0.9, M_u0.5, M_d0.1)
from the full historical M_i series. Results saved to results/exp2_*.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import get_prices
from src.apriori_analysis import (
    full_apriori_analysis,
    compute_runs,
    get_run_counts,
    validate_run_counts,
)

RESULTS_DIR = "results"


def _plot_M_series(M_gold: pd.Series, M_btc: pd.Series) -> str:
    """Plot M_i time series for both assets."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(range(len(M_gold)), M_gold.values, lw=0.7, color="goldenrod")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title("Gold — Daily Rise/Fall Degree M_i")
    axes[0].set_xlabel("Trading Day Index")
    axes[0].set_ylabel("M_i")

    axes[1].plot(range(len(M_btc)), M_btc.values, lw=0.7, color="steelblue")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_title("Bitcoin — Daily Rise/Fall Degree M_i")
    axes[1].set_xlabel("Calendar Day Index")
    axes[1].set_ylabel("M_i")

    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, "exp2_Mi_series.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def _plot_rise_fall_curves(M_gold: pd.Series, M_btc: pd.Series) -> str:
    """Plot separated rise and fall curves for both assets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    for ax_r, ax_f, M, label, color_r, color_f in [
        (axes[0, 0], axes[0, 1], M_gold, "Gold", "green", "red"),
        (axes[1, 0], axes[1, 1], M_btc, "Bitcoin", "limegreen", "tomato"),
    ]:
        idx = np.arange(len(M))
        rise = np.where(M.values > 0, M.values, 0)
        fall = np.where(M.values < 0, M.values, 0)

        ax_r.bar(idx, rise, color=color_r, width=1.0, label="Rise")
        ax_r.set_title(f"{label} — Rise Curve")
        ax_r.set_xlabel("Day Index")
        ax_r.set_ylabel("M_i (positive)")
        ax_r.legend()

        ax_f.bar(idx, fall, color=color_f, width=1.0, label="Fall")
        ax_f.set_title(f"{label} — Fall Curve")
        ax_f.set_xlabel("Day Index")
        ax_f.set_ylabel("M_i (negative)")
        ax_f.legend()

    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, "exp2_rise_fall_curves.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def _plot_run_distributions(
    m_j_gold, n_j_gold, m_j_btc, n_j_btc
) -> str:
    """Bar charts of run-length distributions for both assets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    pairs = [
        (axes[0, 0], m_j_gold, "Gold Rises", "green"),
        (axes[0, 1], n_j_gold, "Gold Falls", "red"),
        (axes[1, 0], m_j_btc, "Bitcoin Rises", "limegreen"),
        (axes[1, 1], n_j_btc, "Bitcoin Falls", "tomato"),
    ]
    for ax, counts, title, color in pairs:
        lengths = sorted(counts.keys())
        freqs = [counts[j] for j in lengths]
        ax.bar(lengths, freqs, color=color, edgecolor="black")
        ax.set_title(f"Run Length Distribution — {title}")
        ax.set_xlabel("Run Length j")
        ax.set_ylabel("Count m_j / n_j")
        ax.set_xticks(lengths)

    plt.tight_layout()
    fpath = os.path.join(RESULTS_DIR, "exp2_run_distributions.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    return fpath


def run_exp2() -> dict:
    """Run Experiment 2: Apriori consecutive rise/fall pattern analysis.

    Returns:
        Dictionary with analysis results for 'gold' and 'btc'.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("  Loading price data...")
    gold_prices = get_prices("GC=F")
    btc_prices = get_prices("BTC-USD")
    print(f"  Gold:    {len(gold_prices)} observations")
    print(f"  Bitcoin: {len(btc_prices)} observations")

    print("  Running Apriori analysis for Gold...")
    gold_res = full_apriori_analysis(gold_prices)

    print("  Running Apriori analysis for Bitcoin...")
    btc_res = full_apriori_analysis(btc_prices)

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    summary = {
        "Asset": ["Gold", "Bitcoin"],
        "u_n_rises": [gold_res["u_n_rises"], btc_res["u_n_rises"]],
        "u_n_falls": [gold_res["u_n_falls"], btc_res["u_n_falls"]],
        "M_u09": [gold_res["M_u09"], btc_res["M_u09"]],
        "M_u05": [gold_res["M_u05"], btc_res["M_u05"]],
        "M_d01": [gold_res["M_d01"], btc_res["M_d01"]],
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "exp2_summary.csv"), index=False)

    print("\n  ---- Apriori Analysis Summary ----")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    for name, res in [("Gold", gold_res), ("Bitcoin", btc_res)]:
        valid = validate_run_counts(res["m_j"], res["n_j"], res["M_series"])
        print(f"  {name} run count validation: {'PASSED' if valid else 'FAILED'}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("  Generating plots...")
    _plot_M_series(gold_res["M_series"], btc_res["M_series"])
    _plot_rise_fall_curves(gold_res["M_series"], btc_res["M_series"])
    _plot_run_distributions(
        gold_res["m_j"], gold_res["n_j"],
        btc_res["m_j"], btc_res["n_j"],
    )

    # ------------------------------------------------------------------ #
    # Run-length distribution tables
    # ------------------------------------------------------------------ #
    for name, res in [("gold", gold_res), ("btc", btc_res)]:
        max_len = max(
            max(res["m_j"].keys(), default=0),
            max(res["n_j"].keys(), default=0),
        )
        rows = []
        for j in range(1, max_len + 1):
            rows.append({"j": j, "m_j (rises)": res["m_j"].get(j, 0), "n_j (falls)": res["n_j"].get(j, 0)})
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, f"exp2_run_lengths_{name}.csv"), index=False
        )

    print("  Experiment 2 complete. Results saved to results/exp2_*")

    return {"gold": gold_res, "btc": btc_res}


if __name__ == "__main__":
    run_exp2()
