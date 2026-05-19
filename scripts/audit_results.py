#!/usr/bin/env python3
"""Audit benchmark_results.csv and benchmark_detail.jsonl for integrity.

Layers 2–4 checks: product correctness, multiplication invariants, depth columns,
timing variance / outlier warnings.

Run from repository root:
    python scripts/audit_results.py
    python scripts/audit_results.py --csv path/to/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def audit_csv(path: Path) -> tuple[int, int]:
    """Return (errors, warnings)."""
    errors = 0
    warnings = 0

    if not path.is_file():
        print(f"ERROR: CSV not found: {path}", file=sys.stderr)
        return 1, 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("ERROR: CSV has no data rows.", file=sys.stderr)
        return 1, 0

    print(f"Auditing {len(rows)} rows from {path}")

    for i, row in enumerate(rows, start=2):
        method = row["method"]
        try:
            a = int(row["operand_a"])
            b = int(row["operand_b"])
            product = int(row["product"])
            digit_width = int(row["digit_width"])
            mults = int(row["multiplications"])
        except (KeyError, ValueError) as exc:
            print(f"ERROR line {i}: bad numeric field: {exc}", file=sys.stderr)
            errors += 1
            continue

        expected = a * b
        if product != expected:
            print(
                f"ERROR line {i}: product {product} != {a}×{b}={expected} "
                f"({method})",
                file=sys.stderr,
            )
            errors += 1

        if method == "vedic":
            want_mults = digit_width**2
            if mults != want_mults:
                print(
                    f"ERROR line {i}: vedic multiplications={mults}, "
                    f"expected {want_mults} (digit_width={digit_width})",
                    file=sys.stderr,
                )
                errors += 1

            depth = compare_depth(digit_width)["vedic"]
            for field, expected_val in (
                ("sequential_depth", depth.sequential_depth),
                ("parallel_width", depth.parallel_width),
                ("parallelism_score", depth.parallelism_score),
            ):
                stored = float(row[field]) if field == "parallelism_score" else int(row[field])
                if field == "parallelism_score":
                    ok = abs(stored - float(expected_val)) < 1e-4
                else:
                    ok = stored == expected_val
                if not ok:
                    print(
                        f"ERROR line {i}: vedic {field}={stored}, expected {expected_val}",
                        file=sys.stderr,
                    )
                    errors += 1

        elif method == "schoolbook":
            want_mults = digit_width**2
            if mults != want_mults:
                print(
                    f"WARN line {i}: schoolbook multiplications={mults}, "
                    f"expected {want_mults} for digit_width={digit_width}",
                    file=sys.stderr,
                )
                warnings += 1

            depth = compare_depth(digit_width)["schoolbook"]
            for field, expected_val in (
                ("sequential_depth", depth.sequential_depth),
                ("parallel_width", depth.parallel_width),
                ("parallelism_score", depth.parallelism_score),
            ):
                stored = float(row[field]) if field == "parallelism_score" else int(row[field])
                if field == "parallelism_score":
                    ok = abs(stored - float(expected_val)) < 1e-4
                else:
                    ok = stored == expected_val
                if not ok:
                    print(
                        f"ERROR line {i}: schoolbook {field}={stored}, expected {expected_val}",
                        file=sys.stderr,
                    )
                    errors += 1

        repeats = _parse_repeat_times(row)
        if repeats and method != "native":
            t_min = min(repeats)
            t_max = max(repeats)
            mean_us = float(row.get("mean_time_us", 0) or 0)
            std_us = float(row.get("std_time_us", 0) or 0)

            if t_min > 0 and t_max > 3 * t_min:
                print(
                    f"WARN line {i}: timing spike max/min={t_max/t_min:.1f}x "
                    f"({method}, pair {row['pair_index']}, width {digit_width})",
                    file=sys.stderr,
                )
                warnings += 1

            if mean_us > 0 and std_us / mean_us > 0.20:
                print(
                    f"WARN line {i}: high CV std/mean={std_us/mean_us:.2%} "
                    f"({method}, pair {row['pair_index']})",
                    file=sys.stderr,
                )
                warnings += 1

    return errors, warnings


def audit_jsonl(path: Path, csv_path: Path) -> int:
    """Light cross-check: line count vs CSV repeat columns."""
    errors = 0
    if not path.is_file():
        print(f"WARN: JSONL not found: {path}", file=sys.stderr)
        return 0

    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        repeat_cols = [c for c in reader.fieldnames or [] if REPEAT_COL.match(c)]
        csv_rows = list(reader)

    expected_jsonl = len(csv_rows) * len(repeat_cols)
    if len(lines) != expected_jsonl:
        print(
            f"WARN: JSONL lines={len(lines)}, expected {expected_jsonl} "
            f"(csv_rows×repeats)",
            file=sys.stderr,
        )
        errors += 0  # warning only

    bad = 0
    for i, line in enumerate(lines, start=1):
        try:
            rec = json.loads(line)
            if rec["operand_a"] * rec["operand_b"] != rec["product"]:
                print(f"ERROR JSONL line {i}: product mismatch", file=sys.stderr)
                bad += 1
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"ERROR JSONL line {i}: {exc}", file=sys.stderr)
            bad += 1

    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit benchmark output files.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = parser.parse_args()

    csv_errors, csv_warnings = audit_csv(args.csv)
    jsonl_errors = audit_jsonl(args.jsonl, args.csv)

    total_errors = csv_errors + jsonl_errors
    print(f"\nSummary: {total_errors} error(s), {csv_warnings} warning(s)")

    if total_errors:
        return 1
    if csv_warnings:
        print("Audit passed with warnings (review timing WARN lines).")
    else:
        print("Audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
