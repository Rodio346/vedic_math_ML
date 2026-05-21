#!/usr/bin/env python3
"""Compare Phase 2B DAG predictions with Phase 3 GPU measurements."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _read_phase3(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_phase2b(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_phase2b_summary(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = int(row["digit_width"])
            out[w] = row
    return out


def _f(row: dict, key: str) -> float:
    val = row.get(key, "")
    if val == "" or val is None:
        return 0.0
    return float(val)


def _i(row: dict, key: str) -> int:
    val = row.get(key, "")
    if val == "" or val is None:
        return 0
    return int(float(val))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2B vs Phase 3 comparison")
    parser.add_argument(
        "--phase3",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "phase3.csv",
    )
    parser.add_argument(
        "--phase2b",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "phase2b"
        / "results"
        / "phase2b.csv",
    )
    parser.add_argument(
        "--phase2b-summary",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "phase2b"
        / "results"
        / "phase2b_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "prediction_vs_actual.csv",
    )
    parser.add_argument(
        "--digit-widths",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9],
    )
    args = parser.parse_args()

    p3_rows = _read_phase3(args.phase3)
    p2_rows = _read_phase2b(args.phase2b)
    p2_summary = _read_phase2b_summary(args.phase2b_summary)

    p3_by_width: dict[int, list[dict]] = defaultdict(list)
    for row in p3_rows:
        p3_by_width[int(row["digit_width"])].append(row)

    p2_vedic: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in p2_rows:
        if row["algorithm"] != "vedic":
            continue
        w = int(row["digit_width"])
        workers = _i(row, "workers")
        if workers == -1:
            continue
        p2_vedic[w][workers].append(_f(row, "completion_time"))

    results: list[dict] = []

    print("Digit width | Predicted crossover | Actual crossover | Error | Valid")
    print("-" * 72)

    valid_count = 0
    total = 0

    for width in args.digit_widths:
        rows_w = p3_by_width.get(width, [])
        vedic_rows = [r for r in rows_w if r["method"] == "vedic"]
        school_rows = [r for r in rows_w if r["method"] == "schoolbook"]

        by_threads_v: dict[int, list[float]] = defaultdict(list)
        for r in vedic_rows:
            by_threads_v[_i(r, "threads_per_column_block")].append(_f(r, "mean_time_us"))

        by_threads_s: dict[int, list[float]] = defaultdict(list)
        for r in school_rows:
            by_threads_s[_i(r, "threads_per_column_block")].append(_f(r, "mean_time_us"))

        mean_v = {t: statistics.mean(v) for t, v in by_threads_v.items() if v}
        mean_s = {t: statistics.mean(v) for t, v in by_threads_s.items() if v}

        serial_v = mean_v.get(1, 0.0)
        vedic_max_actual = 0.0
        if serial_v > 0:
            for t, mv in mean_v.items():
                if mv > 0:
                    vedic_max_actual = max(vedic_max_actual, serial_v / mv)

        p2_w = p2_vedic.get(width, {})
        serial_p2 = statistics.mean(p2_w[1]) if p2_w.get(1) else 0.0
        vedic_max_pred = 0.0
        for workers, times in p2_w.items():
            if workers == 1 or workers == -1:
                continue
            avg_t = statistics.mean(times)
            if avg_t > 0 and serial_p2 > 0:
                vedic_max_pred = max(vedic_max_pred, serial_p2 / avg_t)

        actual_cross: int | None = None
        for t in sorted(mean_v.keys()):
            if t <= 1:
                continue
            sv = mean_s.get(t)
            vv = mean_v.get(t)
            if sv is not None and vv is not None and vv < sv:
                actual_cross = t
                break

        summary_row = p2_summary.get(width, {})
        pred_cross_raw = summary_row.get("avg_crossover_workers", "")
        predicted_cross = float(pred_cross_raw) if pred_cross_raw != "" else None

        error_pct = ""
        accurate = False
        model_valid = False
        if predicted_cross is not None and predicted_cross > 0 and actual_cross is not None:
            error_pct_val = abs(actual_cross - predicted_cross) / predicted_cross * 100.0
            error_pct = f"{error_pct_val:.1f}"
            ratio = actual_cross / predicted_cross
            accurate = 0.5 <= ratio <= 2.0
            if vedic_max_pred > 0 and vedic_max_actual > 0:
                speed_ratio = vedic_max_actual / vedic_max_pred
                model_valid = accurate and 0.5 <= speed_ratio <= 2.0
            else:
                model_valid = accurate
        elif actual_cross is None and (pred_cross_raw == "" or predicted_cross is None):
            accurate = True
            model_valid = True

        if model_valid:
            valid_count += 1
        total += 1

        pred_str = f"{predicted_cross:.2f}" if predicted_cross is not None else "—"
        actual_str = str(actual_cross) if actual_cross is not None else "—"
        err_str = error_pct if error_pct else "—"
        valid_str = "yes" if model_valid else "no"

        print(
            f"     {width:5d}      | {pred_str:>19} | {actual_str:>16} | "
            f"{err_str:>5} | {valid_str:>5}"
        )

        results.append(
            {
                "digit_width": width,
                "predicted_crossover_workers": pred_cross_raw,
                "actual_crossover_threads": actual_cross if actual_cross is not None else "",
                "prediction_error_pct": error_pct,
                "prediction_accurate_within_2x": "yes" if accurate else "no",
                "vedic_max_speedup_actual": round(vedic_max_actual, 6),
                "vedic_max_speedup_predicted": round(vedic_max_pred, 6),
                "model_valid": "yes" if model_valid else "no",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Overall: {valid_count}/{total} digit widths predicted accurately within 2x")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
