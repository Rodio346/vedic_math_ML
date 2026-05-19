#!/usr/bin/env python3
"""Layer 2 — print operation counters for hand-traced reference cases.

Run from repository root:
    python scripts/verify_counters.py

Hand trace for 23×41 (n=2, columns k=0,1,2):
  k=0: 3×1=3           → 1 multiply
  k=1: 2×1 + 3×4 = 14  → 2 multiplies, column sum via add_value_at_position
  k=2: 2×4 + carry      → 1 multiply
  Expected multiplications = 4 (= n²)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vedic_benchmark.algorithms import schoolbook, vedic

TEST_CASES = [(23, 41), (99, 99), (10, 10), (10, 999), (5, 123)]


def main() -> int:
    header = (
        f"{'a':>6} {'b':>6} {'product':>10} "
        f"{'v_m':>5} {'v_a':>5} {'v_c':>5} {'v_tot':>6} "
        f"{'s_m':>5} {'s_a':>5} {'s_c':>5} {'s_tot':>6} {'match':>6}"
    )
    print(header)
    print("-" * len(header))

    failures = 0
    for a, b in TEST_CASES:
        vr, vc = vedic.multiply(a, b)
        sr, sc = schoolbook.multiply(a, b)
        expected = a * b
        match = "OK" if vr == sr == expected else "FAIL"
        if match == "FAIL":
            failures += 1
        print(
            f"{a:>6} {b:>6} {vr:>10} "
            f"{vc.multiplications:>5} {vc.additions:>5} {vc.carry_propagations:>5} {vc.total_ops:>6} "
            f"{sc.multiplications:>5} {sc.additions:>5} {sc.carry_propagations:>5} {sc.total_ops:>6} "
            f"{match:>6}"
        )

    if failures:
        print(f"\n{failures} case(s) FAILED correctness.", file=sys.stderr)
        return 1
    print("\nAll cases OK. Compare v_m/v_a/v_c to your hand trace before trusting benchmarks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
