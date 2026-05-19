"""Theoretical DAG depth and parallelism metrics for multiplication algorithms.

This module models each multiplication algorithm as a directed acyclic graph (DAG)
of single-digit operations, then computes three structural metrics:

  parallel_width    — maximum number of operations that can execute simultaneously
                      (assuming unlimited parallel hardware)
  sequential_depth  — minimum number of sequential steps required even with
                      unlimited parallelism (the critical path length)
  parallelism_score — parallel_width / sequential_depth; higher means the
                      algorithm exposes more parallelism per unit of sequential work

These are *theoretical* metrics derived from the algorithm's structure, not from
measured wall-clock time. They form the bridge between the Python benchmark
(Phase 1) and a future CUDA kernel design (Phase 3): if the parallelism score
scales favourably with digit width, a parallel hardware implementation is justified.

Vedic model (Urdhva-Tiryagbhyam):
  - 2N-1 columns; column k has p = min(k+1, 2N-1-k) independent cross-products.
  - Each column's critical path: 1 (multiply layer) + ceil(log2(p)) (reduction
    tree if p > 1) + 1 (carry layer, only when the column can overflow).
  - parallel_width = N (the middle column, k = N-1, has exactly N cross-products;
    all fire simultaneously on parallel hardware).

Schoolbook model:
  - N partial-product rows; each row is a single pass over the N digits of `a`
    multiplied by one digit of `b`, with an internal carry chain of depth N.
  - Rows are mutually independent, so parallel_width = N * N.
  - After all rows are formed, N-1 sequential accumulation passes merge them.
  - sequential_depth = N (deepest partial-product row) + (N-1) (accumulations)
                     = 2N - 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthMetrics:
    """Theoretical parallelism characteristics for an N-digit multiply."""

    n_digits: int
    algorithm: str
    parallel_width: int
    sequential_depth: int
    parallelism_score: float


def _n(n_digits: int) -> int:
    """Clamp digit count to a minimum of 1."""
    return max(n_digits, 1)


def _column_products(k: int, n: int) -> int:
    """Number of independent cross-products in column k of an N×N multiply.

    For Urdhva-Tiryagbhyam, column k collects all pairs (i, j) where i+j == k
    and 0 <= i, j < n. The count is min(k+1, 2n-1-k).
    """
    return min(k + 1, 2 * n - 1 - k)


def vedic_depth_metrics(n_digits: int) -> DepthMetrics:
    """Compute theoretical DAG metrics for Urdhva-Tiryagbhyam at n_digits.

    Column critical-path breakdown per column k:
      Step 1 — multiply layer:
        All p cross-products fire in parallel → contributes depth 1 regardless
        of p, because on parallel hardware they all execute simultaneously.

      Step 2 — reduction tree (only if p > 1):
        Summing p values via a binary adder tree takes ceil(log2(p)) levels.
        If p == 1 there is nothing to reduce.

      Step 3 — carry layer (only if the column can overflow):
        A column overflows when its maximum possible sum >= 10.
        Worst case: all p products are 9*9 = 81, so max_sum = p * 81.
        If max_sum < 10 (only possible for p=1 and small digits) no carry
        step exists on the critical path.

    The total sequential_depth is the sum of per-column critical paths.
    Columns execute sequentially (each column depends on the carry from the
    previous), so their depths are additive.

    parallel_width = N: the middle column k = N-1 has exactly N cross-products,
    all independent. This is the widest point of the computation.
    """
    n = _n(n_digits)
    parallel_width = n
    sequential_depth = 0

    for k in range(2 * n - 1):
        p = _column_products(k, n)

        # Step 1: multiply layer — always 1 level regardless of p.
        sequential_depth += 1

        # Step 2: reduction tree — only needed when there are multiple products
        # to sum in this column.
        if p > 1:
            sequential_depth += math.ceil(math.log2(p))

        # Step 3: carry layer — only when the column sum can exceed a single
        # digit (i.e. max possible sum >= 10). Worst case: p products of 81.
        if p * 81 >= 10:
            sequential_depth += 1

    parallelism_score = parallel_width / sequential_depth if sequential_depth else 0.0

    return DepthMetrics(
        n_digits=n,
        algorithm="vedic",
        parallel_width=parallel_width,
        sequential_depth=sequential_depth,
        parallelism_score=round(parallelism_score, 4),
    )


def schoolbook_depth_metrics(n_digits: int) -> DepthMetrics:
    """Compute theoretical DAG metrics for schoolbook multiplication at n_digits.

    Schoolbook structure:
      Phase 1 — partial-product rows (N rows, all independent of each other):
        Each row multiplies all N digits of `a` by one digit of `b` with a
        carry chain. The critical path through one row is N steps (N multiplies
        in sequence due to carry propagation). All N rows can execute in
        parallel, so this phase contributes depth N to the critical path.

      Phase 2 — accumulation (N-1 sequential additions):
        The N partial-product rows must be merged into a single result. With
        a naive left-to-right accumulation this takes N-1 sequential passes.
        (A tree-reduction would be ceil(log2(N)), but schoolbook is defined
        as sequential accumulation.)

      sequential_depth = N + (N - 1) = 2N - 1.

      parallel_width = N * N:
        All N*N individual digit multiplications across all rows are mutually
        independent and can execute simultaneously on unlimited hardware.
        This is larger than Vedic's parallel_width of N, but schoolbook's
        sequential_depth grows faster, so its parallelism_score is lower.
    """
    n = _n(n_digits)
    parallel_width = n * n
    sequential_depth = n + (n - 1)  # = 2n - 1; minimum 1 when n == 1

    # Guard against n == 1 giving sequential_depth == 1 (correct, no issue).
    parallelism_score = parallel_width / sequential_depth if sequential_depth else 0.0

    return DepthMetrics(
        n_digits=n,
        algorithm="schoolbook",
        parallel_width=parallel_width,
        sequential_depth=sequential_depth,
        parallelism_score=round(parallelism_score, 4),
    )


def compare_depth(n_digits: int) -> dict[str, DepthMetrics]:
    """Return depth metrics for both algorithms keyed by algorithm name.

    Keys are exactly "vedic" and "schoolbook" — these must match the method
    strings used in runner.py for depth lookup to succeed.
    """
    return {
        "vedic": vedic_depth_metrics(n_digits),
        "schoolbook": schoolbook_depth_metrics(n_digits),
    }