#!/usr/bin/env python3
"""Build verification/counter_parity.json from pairs.json via Python algorithms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vedic_benchmark.algorithms import schoolbook, vedic

DEFAULT_PAIRS_FILE = ROOT / "pairs.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "verification" / "counter_parity.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.pairs_file.read_text(encoding="utf-8"))
    pairs = data["pairs"]

    selected = []
    for width in (2, 3):
        width_pairs = [p for p in pairs if p["digit_width"] == width]
        selected.extend(sorted(width_pairs, key=lambda p: p["pair_index"])[:5])

    entries = []
    for p in selected:
        a, b = p["operand_a"], p["operand_b"]
        for method, fn in (("vedic", vedic.multiply), ("schoolbook", schoolbook.multiply)):
            _, ctr = fn(a, b)
            entries.append(
                {
                    "digit_width": p["digit_width"],
                    "pair_index": p["pair_index"],
                    "operand_a": a,
                    "operand_b": b,
                    "method": method,
                    "multiplications": ctr.multiplications,
                    "additions": ctr.additions,
                    "carry_propagations": ctr.carry_propagations,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
