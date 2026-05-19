#!/usr/bin/env python3
"""Sample N random benchmark CSV rows and print a human-readable cross-check report.

For each sample verifies:
  - product == operand_a * operand_b (and optional live multiply)
  - depth columns match compare_depth(digit_width)
  - vedic multiplications == digit_width ** 2 (when applicable)
  - mean(time_us) from JSONL vs CSV mean_time_us and time_repeat_* columns

Run from repository root:
    python scripts/cross_verify_sample.py
    python scripts/cross_verify_sample.py --count 10 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vedic_benchmark.algorithms import native, schoolbook, vedic
from vedic_benchmark.analysis.depth import compare_depth

DEFAULT_CSV = ROOT / "vedic_benchmark" / "results" / "benchmark_results.csv"
DEFAULT_JSONL = ROOT / "vedic_benchmark" / "results" / "benchmark_detail.jsonl"

REPEAT_COL = re.compile(r"^time_repeat_(\d+)$")


def _parse_repeat_times(row: dict[str, str]) -> list[float]:
    indexed: list[tuple[int, float]] = []
    for key, value in row.items():
        m = REPEAT_COL.match(key)
        if m and value.strip():
            indexed.append((int(m.group(1)), float(value)))
    indexed.sort(key=lambda x: x[0])
    return [t for _, t in indexed]


def _load_jsonl_index(path: Path) -> dict[tuple[str, int, int, str], list[dict]]:
    """Key: (run_id, digit_width, pair_index, method) -> list of records."""
    index: dict[tuple[str, int, int, str], list[dict]] = {}
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (
                rec["run_id"],
                int(rec["digit_width"]),
                int(rec["pair_index"]),
                rec["method"],
            )
            index.setdefault(key, []).append(rec)
    for key in index:
        index[key].sort(key=lambda r: int(r["repeat_index"]))
    return index


def _status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _verify_row(
    row: dict[str, str],
    jsonl_index: dict[tuple[str, int, int, str], list[dict]],
    live_multiply: bool,
) -> tuple[list[str], int]:
    lines: list[str] = []
    issues = 0

    run_id = row["run_id"]
    width = int(row["digit_width"])
    pair_index = int(row["pair_index"])
    method = row["method"]
    a = int(row["operand_a"])
    b = int(row["operand_b"])
    product = int(row["product"])
    expected = a * b

    lines.append(f"  run_id       : {run_id}")
    lines.append(f"  operands     : {a} x {b}")
    lines.append(f"  method       : {method}")

    product_ok = product == expected
    if not product_ok:
        issues += 1
    lines.append(
        f"  product      : {product}  (expected {expected})  [{_status(product_ok)}]"
    )

    if live_multiply:
        if method == "vedic":
            live, _ = vedic.multiply(a, b)
        elif method == "schoolbook":
            live, _ = schoolbook.multiply(a, b)
        else:
            live = native.multiply(a, b)
        live_ok = live == expected
        if not live_ok:
            issues += 1
        lines.append(
            f"  live multiply: {live}  [{_status(live_ok)}]"
        )

    mults = int(row["multiplications"])
    if method == "vedic":
        want_mults = width**2
        mult_ok = mults == want_mults
        if not mult_ok:
            issues += 1
        lines.append(
            f"  multiplications: {mults}  (expected {want_mults} for width {width})  "
            f"[{_status(mult_ok)}]"
        )
    elif method == "schoolbook":
        want_mults = width**2
        mult_ok = mults == want_mults
        note = "" if mult_ok else "  (WARN if operands were mixed-width in benchmark)"
        lines.append(
            f"  multiplications: {mults}  (same-width benchmark expects {want_mults})  "
            f"[{_status(mult_ok)}]{note}"
        )
        if not mult_ok:
            issues += 0  # warn only for schoolbook

    if method in ("vedic", "schoolbook"):
        depth = compare_depth(width)[method]
        for field in ("sequential_depth", "parallel_width", "parallelism_score"):
            stored = float(row[field]) if field == "parallelism_score" else int(row[field])
            expected_val = getattr(depth, field)
            if field == "parallelism_score":
                ok = abs(stored - float(expected_val)) < 1e-4
            else:
                ok = stored == expected_val
            if not ok:
                issues += 1
            lines.append(
                f"  {field}: stored={stored}  theory={expected_val}  [{_status(ok)}]"
            )

    repeats_csv = _parse_repeat_times(row)
    mean_csv = float(row["mean_time_us"])
    std_csv = float(row["std_time_us"])

    key = (run_id, width, pair_index, method)
    jsonl_recs = jsonl_index.get(key, [])
    jsonl_times = [float(r["time_us"]) for r in jsonl_recs]

    if jsonl_times:
        mean_jsonl = statistics.mean(jsonl_times)
        mean_diff = abs(mean_jsonl - mean_csv)
        mean_tol = max(0.05 * mean_csv, 0.01) if mean_csv else 0.01
        mean_ok = mean_diff <= mean_tol
        if not mean_ok:
            issues += 1
        lines.append(
            f"  timing mean  : CSV={mean_csv:.3f} us  JSONL={mean_jsonl:.3f} us  "
            f"diff={mean_diff:.3f}  [{_status(mean_ok)}]"
        )
        if len(jsonl_times) == len(repeats_csv) and repeats_csv:
            pairs_match = all(
                abs(j - c) < 0.02 for j, c in zip(jsonl_times, repeats_csv)
            )
            if not pairs_match:
                issues += 1
            lines.append(
                f"  timing repeats: CSV vs JSONL per-repeat  [{_status(pairs_match)}]"
            )
            lines.append(
                f"                  CSV   : {', '.join(f'{t:.3f}' for t in repeats_csv)}"
            )
            lines.append(
                f"                  JSONL : {', '.join(f'{t:.3f}' for t in jsonl_times)}"
            )
        lines.append(f"  std (CSV)    : {std_csv:.3f} us  (n={len(repeats_csv)} repeats)")
    else:
        lines.append(f"  timing       : no JSONL records for this key  [WARN]")
        if method != "native":
            issues += 1

    lines.append(f"  row verdict  : {'PASS' if issues == 0 else f'{issues} issue(s)'}")
    return lines, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-verify a random sample of benchmark CSV rows.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        metavar="N",
        help="Number of random rows to sample (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip re-running multiply in-process (faster).",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    with args.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("ERROR: CSV has no data rows.", file=sys.stderr)
        return 1

    n = min(args.count, len(rows))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, n)

    print(f"Cross-verify sample: {n} row(s) from {args.csv}")
    print(f"JSONL source: {args.jsonl}")
    print(f"Live multiply: {not args.no_live}\n")

    jsonl_index = _load_jsonl_index(args.jsonl)
    total_issues = 0

    for i, row in enumerate(sample, start=1):
        print("=" * 72)
        print(f"Sample {i}/{n}  (digit_width={row['digit_width']}, "
              f"pair_index={row['pair_index']}, method={row['method']})")
        print("-" * 72)
        block, row_issues = _verify_row(row, jsonl_index, live_multiply=not args.no_live)
        print("\n".join(block))
        total_issues += row_issues
        print()

    print("=" * 72)
    if total_issues == 0:
        print(f"Summary: all {n} sampled row(s) passed cross-checks.")
        return 0
    print(f"Summary: {total_issues}/{n} sampled row(s) reported issue(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
