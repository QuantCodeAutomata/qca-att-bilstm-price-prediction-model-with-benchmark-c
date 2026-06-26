"""
Exponential position-sizing model derived from break-even conditions.

Derives position sizes p_1,...,p_{u_n} by solving iterative break-even
equations (paper Equation 6), then fits s_i = a·e^(b·i) via log-linear
regression, and computes normalised dollar allocations y_i.

# Custom — Context7 found no library equivalent (paper Section 5)
"""
from typing import Dict, Tuple

import numpy as np
from scipy.stats import linregress


def derive_position_sizes(
    u_n: int,
    M_u05: float,
    M_d01: float,
) -> np.ndarray:
    """Derive position sizes from break-even conditions (paper Eq. 6).

    Break-even condition for the n-th position addition:
      p_n · M_u05 + Σ_{k=1}^{n-1} p_k · M_u05^{n-1-k} · (M_d01/u_n)^{n-k} = 0

    Rearranging:
      p_n = -(1/M_u05) · Σ_{k=1}^{n-1} p_k · M_u05^{n-1-k} · (M_d01/u_n)^{n-k}

    Since M_d01 < 0, each term in the sum is negative, so p_n > 0.

    Args:
        u_n: Maximum consecutive fall count (number of positions).
        M_u05: Median of positive M_i values (e.g. 0.0042 for gold).
        M_d01: 10th percentile of negative M_i values (negative, e.g. -0.0080).

    Returns:
        Array of shape (u_n,) with position sizes [p_1, ..., p_{u_n}].

    Raises:
        ValueError: If all positions are not positive.
    """
    ratio = M_d01 / u_n  # negative scalar
    positions = [1.0]  # p_1 = 1.0 (base unit)

    for n in range(2, u_n + 1):
        cumulative = 0.0
        for k in range(1, n):
            power_ratio = n - k          # exponent for ratio = (M_d01/u_n)
            power_Mu = n - 1 - k        # exponent for M_u05
            cumulative += positions[k - 1] * (M_u05 ** power_Mu) * (ratio ** power_ratio)
        p_n = -cumulative / M_u05
        positions.append(p_n)

    positions_arr = np.array(positions, dtype=float)
    if not np.all(positions_arr > 0):
        raise ValueError(
            f"Derived positions contain non-positive values: {positions_arr}. "
            "Check M_u05 and M_d01 parameter signs."
        )
    return positions_arr


def fit_exponential(positions: np.ndarray) -> Tuple[float, float, float]:
    """Fit s_i = a·exp(b·i) via log-linear (OLS) regression.

    log(p_i) = log(a) + b·i  →  OLS on (i, log(p_i)).

    Args:
        positions: Derived position sizes array of shape (u_n,).

    Returns:
        Tuple (a, b, r_squared) where:
          a — scale parameter of the exponential
          b — growth rate
          r_squared — quality of fit (should exceed 0.95)
    """
    i_values = np.arange(1, len(positions) + 1, dtype=float)
    log_p = np.log(positions)
    slope, intercept, r_value, _, _ = linregress(i_values, log_p)
    a = float(np.exp(intercept))
    b = float(slope)
    r_squared = float(r_value ** 2)
    return a, b, r_squared


def compute_allocations(s_values: np.ndarray, PRO: float = 500.0) -> np.ndarray:
    """Normalise fitted exponential values to dollar allocations.

    y_i = (PRO / Σ s_i) · s_i

    Args:
        s_values: Fitted s_i = a·exp(b·i) values.
        PRO: Total available capital in dollars (default 500).

    Returns:
        Dollar allocation array y_i summing exactly to PRO.

    Raises:
        AssertionError: If sum deviates from PRO by more than 1e-6.
    """
    S_total = np.sum(s_values)
    y = (PRO / S_total) * s_values
    assert abs(np.sum(y) - PRO) < 1e-6, (
        f"Allocations sum to {np.sum(y):.6f}, expected {PRO}"
    )
    return y


def full_position_sizing(
    u_n: int,
    M_u05: float,
    M_d01: float,
    PRO: float = 500.0,
) -> Dict:
    """Run the complete position-sizing pipeline.

    Steps:
      1. derive_position_sizes → {p_i}
      2. fit_exponential → a, b, r²
      3. Compute fitted s_i = a·exp(b·i)
      4. compute_allocations → {y_i}

    Args:
        u_n: Maximum consecutive fall count.
        M_u05: Median of positive M_i values.
        M_d01: 10th percentile of negative M_i values (negative).
        PRO: Total capital to allocate (default 500).

    Returns:
        Dictionary with keys:
          'positions': derived p_i array
          's_values': fitted s_i array
          'y_values': dollar allocation y_i array
          'a': exponential scale
          'b': exponential growth rate
          'r_squared': log-linear fit quality
    """
    positions = derive_position_sizes(u_n, M_u05, M_d01)
    a, b, r2 = fit_exponential(positions)
    i_vals = np.arange(1, u_n + 1, dtype=float)
    s_values = a * np.exp(b * i_vals)
    y_values = compute_allocations(s_values, PRO)

    return {
        "positions": positions,
        "s_values": s_values,
        "y_values": y_values,
        "a": a,
        "b": b,
        "r_squared": r2,
    }
