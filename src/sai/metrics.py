"""Compliance metrics and paired bootstrap confidence intervals.

All statistics are computed over seeded episodes. The paired bootstrap
resamples episode indices (B replicates, fixed seed) and reports the
percentile 95% CI of the mean paired difference.
"""
from __future__ import annotations

import numpy as np


def mean_compliance(compliance: list[bool]) -> float:
    return float(np.mean(compliance)) if compliance else 0.0


def paired_bootstrap_ci(
    x: list[float],
    y: list[float],
    n_boot: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Mean(y - x) with percentile bootstrap CI. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.size > 0
    rng = np.random.default_rng(seed)
    diffs = y - x
    boots = np.empty(n_boot)
    n = diffs.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = diffs[idx].mean()
    alpha = 1.0 - ci
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lo, hi


def compliance_curve(results) -> list[float]:
    """Mean compliance per turn across episodes."""
    arr = np.array([r.compliance for r in results], dtype=float)
    return [float(v) for v in arr.mean(axis=0)]


def episode_means(results) -> list[float]:
    return [mean_compliance(r.compliance) for r in results]


def episode_tokens(results) -> list[float]:
    return [float(sum(r.tokens_per_turn)) for r in results]
