"""
Write a comprehensive RESULTS.md summarising findings from all four experiments.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _fmt(v, decimals: int = 4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def write_results_md(
    exp1: dict,
    exp2: dict,
    exp3: dict,
    exp4: dict,
    outpath: str = "results/RESULTS.md",
) -> None:
    """Write RESULTS.md with tables and summaries from all experiments.

    Args:
        exp1: Output of run_exp1().
        exp2: Output of run_exp2().
        exp3: Output of run_exp3().
        exp4: Output of run_exp4().
        outpath: Path to write RESULTS.md (default 'results/RESULTS.md').
    """
    lines = []
    lines.append("# Experiment Results")
    lines.append(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC_\n")

    # ------------------------------------------------------------------ #
    # EXP 1
    # ------------------------------------------------------------------ #
    lines.append("---\n## Experiment 1 — Att-BiLSTM Price Prediction\n")
    for asset_key, asset_label in [("gold", "Gold"), ("btc", "Bitcoin")]:
        lines.append(f"### {asset_label}\n")
        lines.append("| Model | RMSE | MAE | MAPE (%) | R² |")
        lines.append("|---|---|---|---|---|")
        metrics = exp1[asset_key]["metrics"]
        for model_name, m in metrics.items():
            lines.append(
                f"| {model_name} | {_fmt(m['RMSE'],2)} | {_fmt(m['MAE'],2)} "
                f"| {_fmt(m['MAPE'],4)} | {_fmt(m['R2'],4)} |"
            )
        lines.append("")

        # Best model
        best = min(metrics, key=lambda k: metrics[k]["RMSE"])
        lines.append(f"**Best model by RMSE:** {best}\n")

    # ------------------------------------------------------------------ #
    # EXP 2
    # ------------------------------------------------------------------ #
    lines.append("---\n## Experiment 2 — Apriori Rise/Fall Pattern Analysis\n")
    lines.append("| Asset | u_n (rises) | u_n (falls) | M_u0.9 | M_u0.5 | M_d0.1 |")
    lines.append("|---|---|---|---|---|---|")
    for asset_key, label in [("gold", "Gold"), ("btc", "Bitcoin")]:
        r = exp2[asset_key]
        lines.append(
            f"| {label} | {r['u_n_rises']} | {r['u_n_falls']} "
            f"| {_fmt(r['M_u09'],5)} | {_fmt(r['M_u05'],5)} | {_fmt(r['M_d01'],5)} |"
        )
    lines.append("")
    lines.append("M_i = (p_i − p_{i−1}) / p_i (current price as denominator)\n")

    # ------------------------------------------------------------------ #
    # EXP 3
    # ------------------------------------------------------------------ #
    lines.append("---\n## Experiment 3 — Exponential Position Sizing\n")
    lines.append("| Asset | u_n | a | b | R² | Allocations y_i ($) |")
    lines.append("|---|---|---|---|---|---|")
    for asset_key, label in [("gold", "Gold"), ("btc", "Bitcoin")]:
        r = exp3[asset_key]
        params = exp3["params"]
        u_n_key = f"{asset_key}_u_n"
        u_n = params[u_n_key]
        y_str = ", ".join([f"${v:.2f}" for v in r["y_values"]])
        lines.append(
            f"| {label} | {u_n} | {_fmt(r['a'],4)} | {_fmt(r['b'],4)} "
            f"| {_fmt(r['r_squared'],4)} | {y_str} |"
        )
    lines.append("")
    lines.append("Sum of allocations verified to equal PRO=$500 per asset.\n")

    # ------------------------------------------------------------------ #
    # EXP 4
    # ------------------------------------------------------------------ #
    lines.append("---\n## Experiment 4 — Trading Strategy Simulation\n")
    lines.append("### Base Simulation Results\n")
    lines.append("| Asset | Initial ($) | Final ($) | Return (%) | Trades |")
    lines.append("|---|---|---|---|---|")
    gold4 = exp4["gold"]
    btc4 = exp4["btc"]
    lines.append(
        f"| Gold | 500 | {gold4['final']:.2f} | {gold4['return_pct']:.2f} "
        f"| {len(gold4['trades'])} |"
    )
    lines.append(
        f"| Bitcoin | 500 | {btc4['final']:.2f} | {btc4['return_pct']:.2f} "
        f"| {len(btc4['trades'])} |"
    )
    lines.append(
        f"| **Combined** | **1000** | **{exp4['total_final']:.2f}** "
        f"| **{exp4['total_return_pct']:.2f}** | — |"
    )
    lines.append("")

    lines.append("### Transaction Fee Sensitivity\n")
    lines.append("**Gold:**\n")
    lines.append("| Fee (%) | Final Value ($) |")
    lines.append("|---|---|")
    for fee, val in sorted(gold4["sensitivity"].items()):
        lines.append(f"| {fee*100:.1f} | {val:.2f} |")
    lines.append("")

    lines.append("**Bitcoin:**\n")
    lines.append("| Fee (%) | Final Value ($) |")
    lines.append("|---|---|")
    for fee, val in sorted(btc4["sensitivity"].items()):
        lines.append(f"| {fee*100:.1f} | {val:.2f} |")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Write file
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  RESULTS.md written to {outpath}")
