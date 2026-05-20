#!/usr/bin/env python3
"""Run Phase 1 benchmark using operand pairs from pairs.json (Phase 2A ground truth)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark from pairs.json")
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=ROOT / "pairs.json",
    )
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    from vedic_benchmark.benchmark.runner import run_benchmark_from_pairs

    rows, summaries, run_id = run_benchmark_from_pairs(
        pairs_file=args.pairs_file,
        iterations=args.iterations,
        repeat=args.repeat,
        run_id=args.run_id,
        workers=args.workers,
    )
    print(f"Run ID: {run_id}")
    print(f"Rows written: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
