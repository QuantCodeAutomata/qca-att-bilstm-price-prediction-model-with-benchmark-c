"""
Master script: run all four experiments in sequence and save results.

Execution order:
  1. exp_1  — Att-BiLSTM + 7 benchmark models (price prediction)
  2. exp_2  — Apriori run-length analysis (M_i patterns)
  3. exp_3  — Exponential position sizing (break-even derivation)
  4. exp_4  — Full trading strategy simulation (VaR + greedy algorithm)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.run_exp1 import run_exp1
from exp.run_exp2 import run_exp2
from exp.run_exp3 import run_exp3
from exp.run_exp4 import run_exp4
from results.write_results import write_results_md


def main() -> None:
    t0 = time.time()
    os.makedirs("results", exist_ok=True)

    print("=" * 65)
    print("  EXP 1 — Att-BiLSTM + 7 Benchmark Models")
    print("=" * 65)
    exp1_results = run_exp1()

    print("\n" + "=" * 65)
    print("  EXP 2 — Apriori Consecutive Rise/Fall Pattern Analysis")
    print("=" * 65)
    exp2_results = run_exp2()

    # ---- Pass exp_2 parameters to exp_3 ----
    exp3_params = {
        "gold_u_n": exp2_results["gold"]["u_n_falls"],
        "gold_M_u05": exp2_results["gold"]["M_u05"],
        "gold_M_d01": exp2_results["gold"]["M_d01"],
        "btc_u_n": exp2_results["btc"]["u_n_falls"],
        "btc_M_u05": exp2_results["btc"]["M_u05"],
        "btc_M_d01": exp2_results["btc"]["M_d01"],
    }

    print("\n" + "=" * 65)
    print("  EXP 3 — Exponential Position Sizing Model")
    print("=" * 65)
    exp3_results = run_exp3(params=exp3_params)

    print("\n" + "=" * 65)
    print("  EXP 4 — Full Trading Strategy Simulation")
    print("=" * 65)
    exp4_results = run_exp4(
        exp1_results=exp1_results,
        exp3_results=exp3_results,
    )

    print("\n" + "=" * 65)
    print("  Writing RESULTS.md")
    print("=" * 65)
    write_results_md(exp1_results, exp2_results, exp3_results, exp4_results)

    elapsed = time.time() - t0
    print(f"\n  All experiments completed in {elapsed / 60:.1f} minutes.")
    print("  Artifacts saved under results/")


if __name__ == "__main__":
    main()
