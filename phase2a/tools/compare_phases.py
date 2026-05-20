#!/usr/bin/env python3
"""Compare Phase 1 Python CSV vs Phase 2A C++ CSV by digit width."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHODS = ("vedic", "schoolbook")
MIN_PAIRS_WARN = 10
NOISE_CV_THRESHOLD = 0.20


def platform_slug(platform: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", platform.strip().lower())
    return slug.strip("_") or "unknown"


def latest_run_id(rows: list[dict[str, str]], column: str = "run_id") -> str | None:
    ids = {row[column] for row in rows if row.get(column)}
    return max(ids) if ids else None


def filter_latest_run(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str | None]:
    run_id = latest_run_id(rows)
    if run_id is None:
        return rows, None
    return [row for row in rows if row.get("run_id") == run_id], run_id


def load_phase1_stats(
    path: Path,
) -> tuple[dict[tuple[int, str], float], dict[int, int], str | None]:
    with path.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))

    rows, run_id = filter_latest_run(raw)
    if run_id and len(rows) < len(raw):
        print(
            f"Phase 1: using latest run_id={run_id} "
            f"({len(rows)} rows, dropped {len(raw) - len(rows)} older)",
            file=sys.stderr,
        )
    elif run_id:
        print(f"Phase 1: run_id={run_id} ({len(rows)} rows)", file=sys.stderr)
    else:
        print("Phase 1: no run_id column; using all rows", file=sys.stderr)

    times: dict[tuple[int, str], list[float]] = defaultdict(list)
    pair_keys: dict[int, set[int]] = defaultdict(set)

    for row in rows:
        if row["method"] not in METHODS:
            continue
        width = int(row["digit_width"])
        times[(width, row["method"])].append(float(row["mean_time_us"]))
        pair_keys[width].add(int(row["pair_index"]))

    means = {key: statistics.mean(vals) for key, vals in times.items()}
    return means, {w: len(ix) for w, ix in pair_keys.items()}, run_id


def load_phase2_stats(
    path: Path,
) -> tuple[
    dict[tuple[int, str, str], float],
    dict[tuple[int, str, str], float],
    dict[int, dict[str, int]],
    str | None,
    list[str],
]:
    with path.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))

    rows, run_id = filter_latest_run(raw)
    if run_id and len(rows) < len(raw):
        print(
            f"Phase 2A: using latest run_id={run_id} "
            f"({len(rows)} rows, dropped {len(raw) - len(rows)} older)",
            file=sys.stderr,
        )
    elif run_id:
        print(f"Phase 2A: run_id={run_id} ({len(rows)} rows)", file=sys.stderr)
    else:
        print("Phase 2A: no run_id column; using all rows", file=sys.stderr)

    means: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    stds: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    pair_keys: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for row in rows:
        if row["method"] not in METHODS:
            continue
        width = int(row["digit_width"])
        plat = platform_slug(row.get("platform", "unknown"))
        key = (width, row["method"], plat)
        means[key].append(float(row["mean_time_us"]))
        stds[key].append(float(row["std_time_us"]))
        pair_keys[width][plat].add(int(row["pair_index"]))

    mean_by_key = {k: statistics.mean(v) for k, v in means.items()}
    avg_std_by_key = {k: statistics.mean(v) for k, v in stds.items()}
    pair_counts = {
        w: {p: len(ix) for p, ix in plat_map.items()} for w, plat_map in pair_keys.items()
    }
    platforms = sorted({p for _, _, p in mean_by_key})
    return mean_by_key, avg_std_by_key, pair_counts, run_id, platforms


def vs_ratio(school_us: float, vedic_us: float) -> float:
    return school_us / vedic_us if vedic_us else 0.0


def cpp_noise_at_width(
    width: int,
    platform: str,
    cpp_std: dict[tuple[int, str, str], float],
    cpp_means: dict[tuple[int, str, str], float],
) -> bool:
    for method in METHODS:
        mean_us = cpp_means.get((width, method, platform), 0.0)
        std_us = cpp_std.get((width, method, platform), 0.0)
        if mean_us > 0 and std_us / mean_us > NOISE_CV_THRESHOLD:
            return True
    return False


def warn_pair_counts_py(pair_counts: dict[int, int], widths: list[int]) -> None:
    for width in widths:
        count = pair_counts.get(width, 0)
        if count < MIN_PAIRS_WARN:
            print(
                f"WARNING: Phase 1 digit_width={width} has only {count} pairs "
                f"(expected >= {MIN_PAIRS_WARN})",
                file=sys.stderr,
            )


def warn_pair_counts_cpp(
    pair_counts: dict[int, dict[str, int]], widths: list[int], platforms: list[str]
) -> None:
    for width in widths:
        for plat in platforms:
            count = pair_counts.get(width, {}).get(plat, 0)
            if count < MIN_PAIRS_WARN:
                print(
                    f"WARNING: Phase 2A digit_width={width} platform={plat} "
                    f"has only {count} pairs (expected >= {MIN_PAIRS_WARN})",
                    file=sys.stderr,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py-csv", "--phase1", dest="py_csv", type=Path)
    parser.add_argument("--cpp-csv", "--phase2", dest="cpp_csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "phase2a" / "results" / "comparison.csv",
    )
    parser.set_defaults(
        py_csv=ROOT / "vedic_benchmark" / "results" / "benchmark_results.csv",
        cpp_csv=ROOT / "phase2a" / "results" / "phase2a.csv",
    )
    args = parser.parse_args()

    if not args.py_csv.is_file():
        print(f"Missing Python CSV: {args.py_csv}", file=sys.stderr)
        return 1
    if not args.cpp_csv.is_file():
        print(f"Missing C++ CSV: {args.cpp_csv}", file=sys.stderr)
        return 1

    py_means, py_pairs, _ = load_phase1_stats(args.py_csv)
    cpp_means, cpp_std, cpp_pairs, _, platforms = load_phase2_stats(args.cpp_csv)

    widths = sorted({w for w, _ in py_means} & {w for w, _, _ in cpp_means})
    only_cpp = sorted({w for w, _, _ in cpp_means} - {w for w, _ in py_means})
    only_py = sorted({w for w, _ in py_means} - {w for w, _, _ in cpp_means})
    if only_cpp:
        print(f"Note: widths only in C++ CSV: {only_cpp}", file=sys.stderr)
    if only_py:
        print(f"Note: widths only in Python CSV: {only_py}", file=sys.stderr)

    warn_pair_counts_py(py_pairs, widths)
    warn_pair_counts_cpp(cpp_pairs, widths, platforms)

    primary = platforms[0] if platforms else "unknown"

    fieldnames = [
        "digit_width",
        "vedic_py",
        "vedic_cpp",
        "school_py",
        "school_cpp",
        "vs_ratio_py",
        "vs_ratio_cpp",
        "ratio_change",
        "platform",
        "variance_flag",
    ]
    for plat in platforms:
        if plat != primary:
            fieldnames.append(f"vs_ratio_cpp_{plat}")

    rows_out: list[dict[str, str]] = []

    hdr = (
        f"{'W':>3} {'!':>1}  {'vedic_py':>9} {'vedic_cpp':>9} "
        f"{'school_py':>9} {'school_cpp':>9} {'vs_py':>7} {'vs_cpp':>7} {'d_vs':>7}"
    )
    for plat in platforms:
        if plat != primary:
            hdr += f"  {'vs_'+plat:>10}"
    hdr += f"  {'platform':>8}"
    print(hdr)
    print("-" * len(hdr))

    for w in widths:
        v_py = py_means.get((w, "vedic"), 0.0)
        s_py = py_means.get((w, "schoolbook"), 0.0)
        vs_py = vs_ratio(s_py, v_py)

        v_cpp = cpp_means.get((w, "vedic", primary), 0.0)
        s_cpp = cpp_means.get((w, "schoolbook", primary), 0.0)
        vs_cpp = vs_ratio(s_cpp, v_cpp)
        ratio_change = vs_cpp - vs_py

        flag = "!" if cpp_noise_at_width(w, primary, cpp_std, cpp_means) else ""
        for plat in platforms:
            if plat != primary and cpp_noise_at_width(w, plat, cpp_std, cpp_means):
                flag = "!"

        row: dict[str, str] = {
            "digit_width": str(w),
            "vedic_py": f"{v_py:.3f}",
            "vedic_cpp": f"{v_cpp:.3f}",
            "school_py": f"{s_py:.3f}",
            "school_cpp": f"{s_cpp:.3f}",
            "vs_ratio_py": f"{vs_py:.4f}",
            "vs_ratio_cpp": f"{vs_cpp:.4f}",
            "ratio_change": f"{ratio_change:.4f}",
            "platform": primary,
            "variance_flag": flag,
        }
        for plat in platforms:
            if plat == primary:
                continue
            s_p = cpp_means.get((w, "schoolbook", plat), 0.0)
            v_p = cpp_means.get((w, "vedic", plat), 0.0)
            row[f"vs_ratio_cpp_{plat}"] = f"{vs_ratio(s_p, v_p):.4f}"
            row[f"vedic_cpp_{plat}"] = f"{v_p:.3f}"
            row[f"school_cpp_{plat}"] = f"{s_p:.3f}"

        line = (
            f"{w:>3}{flag:>1}  {v_py:>9.2f} {v_cpp:>9.2f} "
            f"{s_py:>9.2f} {s_cpp:>9.2f} {vs_py:>7.4f} {vs_cpp:>7.4f} {ratio_change:>+7.4f}"
        )
        for plat in platforms:
            if plat != primary:
                line += f"  {float(row[f'vs_ratio_cpp_{plat}']):>10.4f}"
        line += f"  {primary:>8}"
        print(line)
        rows_out.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
