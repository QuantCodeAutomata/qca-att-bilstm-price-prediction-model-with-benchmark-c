"""
Apriori-inspired run-length analysis for consecutive rise/fall patterns.

Computes M_i series run statistics and derives threshold parameters
(u_n, M_u0.9, M_u0.5, M_d0.1) used in the position sizing model.

# Custom — Context7 found no library equivalent (paper Section 4)
"""
import itertools
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_runs(M_series: pd.Series) -> List[Tuple[int, int]]:
    """Identify consecutive rise/fall runs in the M_i series.

    Uses itertools.groupby on the sign of each M_i value.
    Zero-valued M_i days are excluded from both rise and fall runs.

    Args:
        M_series: pd.Series of M_i values.

    Returns:
        List of (sign, length) tuples where sign ∈ {+1, -1}.
    """
    signs = np.sign(M_series.values)
    runs = []
    for key, group in itertools.groupby(signs):
        length = sum(1 for _ in group)
        if key != 0:
            runs.append((int(key), length))
    return runs


def get_run_counts(runs: List[Tuple[int, int]], sign: int) -> Counter:
    """Count occurrences of each run length for a given direction.

    Args:
        runs: List of (sign, length) tuples from compute_runs().
        sign: +1 for rises, -1 for falls.

    Returns:
        Counter mapping run_length → occurrence_count.
    """
    lengths = [r[1] for r in runs if r[0] == sign]
    return Counter(lengths)


def compute_u_n(run_counts: Counter, threshold: float = 0.90) -> int:
    """Find minimum j such that ≥ threshold fraction of runs have length ≤ j.

    Apriori minimum support: the smallest j covering `threshold`
    (default 90%) of all observed consecutive sequences.

    Args:
        run_counts: Counter from get_run_counts().
        threshold: Cumulative proportion threshold (default 0.90).

    Returns:
        Integer u_n.
    """
    total = sum(run_counts.values())
    if total == 0:
        return 1
    cumulative = 0
    for j in sorted(run_counts.keys()):
        cumulative += run_counts[j]
        if cumulative / total >= threshold:
            return j
    return max(run_counts.keys())


def compute_percentile_stats(M_series: pd.Series) -> Dict[str, float]:
    """Compute M_u0.9, M_u0.5, M_d0.1 from positive and negative M_i values.

    Args:
        M_series: pd.Series of M_i values.

    Returns:
        Dictionary with keys 'M_u09', 'M_u05', 'M_d01'.
          M_u09 — 90th percentile of positive (rise) M_i values
          M_u05 — 50th percentile (median) of positive M_i values
          M_d01 — 10th percentile of negative (fall) M_i values (most negative 10%)
    """
    rise_vals = M_series[M_series > 0].values
    fall_vals = M_series[M_series < 0].values
    return {
        "M_u09": float(np.percentile(rise_vals, 90)),
        "M_u05": float(np.percentile(rise_vals, 50)),
        "M_d01": float(np.percentile(fall_vals, 10)),
    }


def validate_run_counts(
    m_j: Counter,
    n_j: Counter,
    M_series: pd.Series,
    tolerance: int = 5,
) -> bool:
    """Validate that run counts account for all non-zero M_i observations.

    Σ_j (m_j × j) + Σ_j (n_j × j) ≈ total number of non-zero M_i.

    Args:
        m_j: Rise run counts (Counter).
        n_j: Fall run counts (Counter).
        M_series: Full M_i series.
        tolerance: Allowed discrepancy (default 5).

    Returns:
        True if validation passes.

    Raises:
        AssertionError if discrepancy exceeds tolerance.
    """
    total_rise_days = sum(j * m_j[j] for j in m_j)
    total_fall_days = sum(j * n_j[j] for j in n_j)
    total_nonzero = int((M_series != 0).sum())
    discrepancy = abs(total_rise_days + total_fall_days - total_nonzero)
    assert discrepancy <= tolerance, (
        f"Run counts do not match: rise_days={total_rise_days}, "
        f"fall_days={total_fall_days}, total_nonzero={total_nonzero}, "
        f"discrepancy={discrepancy}"
    )
    return True


def full_apriori_analysis(prices: pd.Series) -> Dict:
    """Run complete Apriori run-length analysis on a price series.

    Steps:
      1. Compute M_i series.
      2. Identify all consecutive runs.
      3. Separate into rise/fall run-length distributions.
      4. Determine u_n at 90% support threshold for both directions.
      5. Compute M_u0.9, M_u0.5, M_d0.1.
      6. Validate run counts.

    Args:
        prices: pd.Series of daily closing prices.

    Returns:
        Dictionary with keys:
          'M_series', 'runs', 'm_j', 'n_j',
          'u_n_rises', 'u_n_falls', 'M_u09', 'M_u05', 'M_d01'
    """
    M = (prices - prices.shift(1)) / prices
    M = M.dropna()

    runs = compute_runs(M)
    m_j = get_run_counts(runs, sign=+1)
    n_j = get_run_counts(runs, sign=-1)

    u_n_rises = compute_u_n(m_j, threshold=0.90)
    u_n_falls = compute_u_n(n_j, threshold=0.90)

    stats = compute_percentile_stats(M)

    validate_run_counts(m_j, n_j, M)

    return {
        "M_series": M,
        "runs": runs,
        "m_j": m_j,
        "n_j": n_j,
        "u_n_rises": u_n_rises,
        "u_n_falls": u_n_falls,
        "M_u09": stats["M_u09"],
        "M_u05": stats["M_u05"],
        "M_d01": stats["M_d01"],
    }
