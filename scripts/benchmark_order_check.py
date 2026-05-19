#!/usr/bin/env python3
"""Layer 4 — check whether method run order affects relative timing.

Compares schoolbook/vedic mean time ratio for pair (79, 99) under two orderings.
Large ratio shifts may indicate cache-warming bias.

Run from repository root:
    python scripts/benchmark_order_check.py
    python scripts/benchmark_order_check.py --iterations 5000 --repeat 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vedic_benchmark.benchmark.runner import _TIMERS

DEFAULT_A = 79
DEFAULT_B = 99


def _mean_time_us(method: str, a: int, b: int, iterations: int, repeat: int) -> float:
    fn = _TIMERS[method]
    timer = timeit.Timer(lambda: fn(a, b))
    samples = timer.repeat(repeat=repeat, number=iterations)
    per_call = [(t / iterations) * 1_000_000 for t in samples]
    return statistics.mean(per_call)


def _run_order(
    order: tuple[str, ...],
    a: int,
    b: int,
    iterations: int,
    repeat: int,
) -> dict[str, float]:
    times: dict[str, float] = {}
    for method in order:
        times[method] = _mean_time_us(method, a, b, iterations, repeat)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description="Method order timing bias check.")
    parser.add_argument("--a", type=int, default=DEFAULT_A)
    parser.add_argument("--b", type=int, default=DEFAULT_B)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    order_a = ("vedic", "schoolbook", "native")
    order_b = ("schoolbook", "vedic", "native")

    print(f"Pair ({args.a}, {args.b}), iterations={args.iterations}, repeat={args.repeat}\n")

    t_a = _run_order(order_a, args.a, args.b, args.iterations, args.repeat)
    t_b = _run_order(order_b, args.a, args.b, args.iterations, args.repeat)

    ratio_a = t_a["schoolbook"] / t_a["vedic"] if t_a["vedic"] else 0
    ratio_b = t_b["schoolbook"] / t_b["vedic"] if t_b["vedic"] else 0

    print(f"Order {order_a}:")
    print(f"  vedic={t_a['vedic']:.2f} µs  schoolbook={t_a['schoolbook']:.2f} µs  ratio={ratio_a:.3f}")
    print(f"Order {order_b}:")
    print(f"  vedic={t_b['vedic']:.2f} µs  schoolbook={t_b['schoolbook']:.2f} µs  ratio={ratio_b:.3f}")

    if ratio_a > 0:
        pct_diff = abs(ratio_b - ratio_a) / ratio_a * 100
        print(f"\nRatio shift: {pct_diff:.1f}%")
        if pct_diff > 10:
            print(
                "WARN: >10% shift — consider randomizing method order or "
                "warming up each method before timing.",
                file=sys.stderr,
            )
            return 0
    print("\nOrder sensitivity within 10% — acceptable for structural comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
