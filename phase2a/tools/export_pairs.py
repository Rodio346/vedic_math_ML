#!/usr/bin/env python3
"""Export operand pairs for Phase 1 and Phase 2A using identical RNG to Python runner.

Replicates: random.Random(seed + digit_width).randint(low, high)
from vedic_benchmark.benchmark.runner._random_pairs logic.

Run from repository root:
    python phase2a/tools/export_pairs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9]
DEFAULT_PAIRS = 20
DEFAULT_SEED = 42


def digit_range(n_digits: int) -> tuple[int, int]:
    low = 10 ** (n_digits - 1)
    high = 10**n_digits - 1
    return low, high


def generate_pairs(
    digit_widths: list[int],
    pairs_per_width: int,
    seed: int,
) -> list[dict]:
    entries: list[dict] = []
    for width in digit_widths:
        rng = random.Random(seed + width)
        low, high = digit_range(width)
        for pair_index in range(pairs_per_width):
            entries.append(
                {
                    "digit_width": width,
                    "pair_index": pair_index,
                    "operand_a": rng.randint(low, high),
                    "operand_b": rng.randint(low, high),
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Export pairs.json ground truth.")
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=DEFAULT_WIDTHS,
    )
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pairs.json"),
        help="Output path (default: repo root pairs.json)",
    )
    args = parser.parse_args()

    payload = {
        "seed": args.seed,
        "pairs_per_width": args.pairs,
        "digit_widths": args.widths,
        "pairs": generate_pairs(args.widths, args.pairs, args.seed),
    }

    text = json.dumps(payload, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"Wrote {len(payload['pairs'])} pairs to {args.output.resolve()}")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
