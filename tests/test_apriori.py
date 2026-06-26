"""
Tests for src/apriori_analysis.py — run detection, u_n calculation,
percentile statistics, and validation.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.apriori_analysis import (
    compute_runs,
    get_run_counts,
    compute_u_n,
    compute_percentile_stats,
    validate_run_counts,
    full_apriori_analysis,
)


@pytest.fixture
def simple_M():
    """M_i series with known runs: +,+,+,-,-,+,+,-,-,-"""
    vals = [0.01, 0.02, 0.03, -0.01, -0.02, 0.04, 0.01, -0.03, -0.01, -0.005]
    return pd.Series(vals)


@pytest.fixture
def random_prices():
    rng = np.random.RandomState(99)
    p = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, 600))
    return pd.Series(p, index=pd.date_range("2018-01-01", periods=600, freq="B"))


class TestComputeRuns:
    def test_known_runs(self, simple_M):
        runs = compute_runs(simple_M)
        # Expected: (+1,3), (-1,2), (+1,2), (-1,3)
        assert runs == [(1, 3), (-1, 2), (1, 2), (-1, 3)]

    def test_all_rises(self):
        M = pd.Series([0.01, 0.02, 0.03])
        runs = compute_runs(M)
        assert runs == [(1, 3)]

    def test_all_falls(self):
        M = pd.Series([-0.01, -0.02, -0.03])
        runs = compute_runs(M)
        assert runs == [(-1, 3)]

    def test_alternating(self):
        M = pd.Series([0.01, -0.01, 0.02, -0.02])
        runs = compute_runs(M)
        assert len(runs) == 4
        assert all(r[1] == 1 for r in runs)

    def test_zeros_excluded(self):
        M = pd.Series([0.0, 0.0, 0.01, -0.01])
        runs = compute_runs(M)
        # Zeros should not create (+1,2) or (-1,2) runs
        assert all(r[0] != 0 for r in runs)

    def test_single_observation(self):
        M = pd.Series([0.05])
        runs = compute_runs(M)
        assert runs == [(1, 1)]


class TestGetRunCounts:
    def test_rise_counts(self, simple_M):
        runs = compute_runs(simple_M)
        m_j = get_run_counts(runs, sign=+1)
        # Two rise runs: length 3 and length 2
        assert m_j[3] == 1
        assert m_j[2] == 1
        assert m_j.total() == 2

    def test_fall_counts(self, simple_M):
        runs = compute_runs(simple_M)
        n_j = get_run_counts(runs, sign=-1)
        assert n_j[2] == 1
        assert n_j[3] == 1

    def test_empty_series(self):
        counts = get_run_counts([], sign=+1)
        assert counts.total() == 0


class TestComputeUn:
    def test_simple_90_percent(self):
        # Counts: {1: 7, 2: 2, 3: 1} total=10
        # cum(1)=0.7, cum(2)=0.9 → u_n should be 2
        counts = Counter({1: 7, 2: 2, 3: 1})
        assert compute_u_n(counts, threshold=0.90) == 2

    def test_all_length_one(self):
        counts = Counter({1: 100})
        assert compute_u_n(counts, threshold=0.90) == 1

    def test_exact_threshold(self):
        counts = Counter({1: 9, 2: 1})  # cum(1) = 0.90
        assert compute_u_n(counts, threshold=0.90) == 1

    def test_empty_counter(self):
        # Should not raise, return 1 (default)
        assert compute_u_n(Counter(), threshold=0.90) == 1

    def test_high_threshold(self):
        counts = Counter({1: 5, 2: 3, 3: 2})
        u = compute_u_n(counts, threshold=1.0)
        assert u == 3

    def test_u_n_positive(self, random_prices):
        M = (random_prices - random_prices.shift(1)) / random_prices
        M = M.dropna()
        from src.apriori_analysis import compute_runs, get_run_counts
        runs = compute_runs(M)
        n_j = get_run_counts(runs, -1)
        u_n = compute_u_n(n_j, 0.90)
        assert u_n >= 1


class TestPercentileStats:
    def test_keys_present(self, simple_M):
        stats = compute_percentile_stats(simple_M)
        assert "M_u09" in stats
        assert "M_u05" in stats
        assert "M_d01" in stats

    def test_M_u09_ge_M_u05(self, simple_M):
        stats = compute_percentile_stats(simple_M)
        assert stats["M_u09"] >= stats["M_u05"]

    def test_M_u05_positive(self, simple_M):
        stats = compute_percentile_stats(simple_M)
        assert stats["M_u05"] > 0

    def test_M_d01_negative(self, simple_M):
        stats = compute_percentile_stats(simple_M)
        assert stats["M_d01"] < 0

    def test_realistic_gold_range(self):
        """M_u0.5 for gold should be approx 0.0041-0.0043 per the paper."""
        rng = np.random.RandomState(0)
        # Simulate ~1600 daily observations with typical gold volatility
        M_vals = rng.normal(0.0003, 0.008, 1600)
        M = pd.Series(M_vals)
        stats = compute_percentile_stats(M)
        # Should be a small positive number
        assert 0.001 < stats["M_u05"] < 0.02


class TestValidateRunCounts:
    def test_valid_series(self, simple_M):
        runs = compute_runs(simple_M)
        m_j = get_run_counts(runs, +1)
        n_j = get_run_counts(runs, -1)
        assert validate_run_counts(m_j, n_j, simple_M)

    def test_fails_on_mismatch(self):
        M = pd.Series([0.01, 0.02, -0.01, -0.02])
        runs = compute_runs(M)
        m_j = get_run_counts(runs, +1)
        n_j = get_run_counts(runs, -1)
        # Corrupt counts
        m_j_corrupt = Counter({1: 999})
        with pytest.raises(AssertionError):
            validate_run_counts(m_j_corrupt, n_j, M, tolerance=1)


class TestFullAprioriAnalysis:
    def test_output_keys(self, random_prices):
        result = full_apriori_analysis(random_prices)
        expected_keys = {
            "M_series", "runs", "m_j", "n_j",
            "u_n_rises", "u_n_falls", "M_u09", "M_u05", "M_d01",
        }
        assert expected_keys == set(result.keys())

    def test_u_n_at_least_1(self, random_prices):
        result = full_apriori_analysis(random_prices)
        assert result["u_n_rises"] >= 1
        assert result["u_n_falls"] >= 1

    def test_M_values_sign(self, random_prices):
        result = full_apriori_analysis(random_prices)
        assert result["M_u05"] > 0
        assert result["M_d01"] < 0
