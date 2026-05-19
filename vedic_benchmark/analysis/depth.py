"""Theoretical DAG depth and parallelism metrics for multiplication algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthMetrics:
    """Theoretical parallelism characteristics for an n-digit multiply."""

    n_digits: int
    algorithm: str
    parallel_width: int
    sequential_depth: int
    parallelism_score: float


def _n(n_digits: int) -> int:
    return max(n_digits, 1)


def _column_products(k: int, n: int) -> int:
    """Count of i+j=k terms for n-digit operands (0-indexed)."""
    return min(k + 1, 2 * n - 1 - k)


def vedic_depth_metrics(n_digits: int) -> DepthMetrics:
    """Theoretical DAG metrics for Urdhva-Tiryagbhyam.

    parallel_width = max_k min(k+1, 2N-1-k) = N
    sequential_depth = sum_k (1 mult layer + ceil(log2(p)) if p>1 + 1 carry)
    """
    n = _n(n_digits)
    parallel_width = n
    sequential_depth = 0
    for k in range(2 * n - 1):
        p = _column_products(k, n)
        sequential_depth += 1
        if p > 1:
            sequential_depth += math.ceil(math.log2(p))
        sequential_depth += 1
    parallelism_score = parallel_width / sequential_depth
    return DepthMetrics(
        n_digits=n,
        algorithm="vedic",
        parallel_width=parallel_width,
        sequential_depth=sequential_depth,
        parallelism_score=round(parallelism_score, 4),
    )


def schoolbook_depth_metrics(n_digits: int) -> DepthMetrics:
    """Theoretical DAG metrics for schoolbook multiplication.

    parallel_width = N
    sequential_depth = N + (N-1)*N
    """
    n = _n(n_digits)
    parallel_width = n
    sequential_depth = n + (n - 1) * n
    parallelism_score = parallel_width / sequential_depth
    return DepthMetrics(
        n_digits=n,
        algorithm="schoolbook",
        parallel_width=parallel_width,
        sequential_depth=sequential_depth,
        parallelism_score=round(parallelism_score, 4),
    )


def compare_depth(n_digits: int) -> dict[str, DepthMetrics]:
    """Return depth metrics for both algorithms at a given digit width."""
    return {
        "vedic": vedic_depth_metrics(n_digits),
        "schoolbook": schoolbook_depth_metrics(n_digits),
    }
