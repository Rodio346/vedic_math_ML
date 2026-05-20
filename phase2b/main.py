"""Phase 2B CLI: DAG scheduling and crossover analysis over pairs.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phase2b.analysis.crossover import find_crossover
from phase2b.dag.scheduler import schedule_all_workers
from phase2b.dag.schoolbook_dag import build_schoolbook_dag
from phase2b.dag.vedic_dag import build_vedic_dag
from vedic_benchmark.algorithms._digits import int_to_digits


CSV_COLUMNS = [
    "digit_width",
    "pair_index",
    "operand_a",
    "operand_b",
    "algorithm",
    "workers",
    "completion_time",
    "theoretical_min",
    "utilisation",
    "speedup_vs_serial",
    "crossover_workers",
    "efficiency_ratio",
]

SUMMARY_COLUMNS = [
    "digit_width",
    "avg_crossover_workers",
    "pct_pairs_with_crossover",
    "avg_efficiency_ratio",
    "vedic_avg_min_completion",
    "school_avg_min_completion",
    "avg_critical_path_ratio",
]


def _padded_n(a: int, b: int) -> int:
    return max(len(int_to_digits(a)), len(int_to_digits(b)))


def _workers_csv_value(workers: int | float) -> int:
    """CSV encoding: -1 = unlimited parallel workers (critical path)."""
    return -1 if workers == math.inf else int(workers)


def _ensure_tests_pass() -> None:
    phase2b_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=phase2b_dir,
        check=False,
    )
    if result.returncode != 0:
        print("Refusing to run: pytest tests/ failed. Fix tests before main.py.", file=sys.stderr)
        sys.exit(1)


def run_benchmark(
    pairs_path: Path,
    digit_widths: list[int],
    output_path: Path,
    summary_path: Path,
    *,
    skip_test_gate: bool = False,
) -> None:
    if not skip_test_gate:
        _ensure_tests_pass()

    with pairs_path.open(encoding="utf-8") as f:
        data = json.load(f)

    pairs = [p for p in data["pairs"] if p["digit_width"] in digit_widths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    width_stats: dict[int, list[dict]] = defaultdict(list)

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for pair in pairs:
            a = pair["operand_a"]
            b = pair["operand_b"]
            n = _padded_n(a, b)

            v_nodes, v_topo = build_vedic_dag(a, b)
            s_nodes, s_topo = build_schoolbook_dag(a, b)
            v_sched = schedule_all_workers(v_nodes, v_topo, n)
            s_sched = schedule_all_workers(s_nodes, s_topo, n)
            crossover = find_crossover(v_sched, s_sched)

            width_stats[pair["digit_width"]].append(
                {
                    "crossover": crossover.crossover_workers,
                    "efficiency_ratio": crossover.efficiency_ratio,
                    "vedic_min": crossover.vedic_min_completion,
                    "school_min": crossover.school_min_completion,
                }
            )

            for algorithm, sched in (("vedic", v_sched), ("schoolbook", s_sched)):
                serial = sched[1].completion_time
                theoretical_min = sched[math.inf].completion_time

                for workers, result in sorted(
                    sched.items(),
                    key=lambda item: (item[0] == math.inf, item[0]),
                ):
                    w_csv = _workers_csv_value(workers)
                    speedup = (
                        serial / result.completion_time
                        if result.completion_time
                        else 0.0
                    )
                    writer.writerow(
                        {
                            "digit_width": pair["digit_width"],
                            "pair_index": pair["pair_index"],
                            "operand_a": a,
                            "operand_b": b,
                            "algorithm": algorithm,
                            "workers": w_csv,
                            "completion_time": result.completion_time,
                            "theoretical_min": theoretical_min,
                            "utilisation": round(result.utilisation, 6),
                            "speedup_vs_serial": round(speedup, 6),
                            "crossover_workers": crossover.crossover_workers
                            if crossover.crossover_workers is not None
                            else "",
                            "efficiency_ratio": round(crossover.efficiency_ratio, 6),
                        }
                    )

    with summary_path.open("w", newline="", encoding="utf-8") as sum_f:
        writer = csv.DictWriter(sum_f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        for width in sorted(width_stats.keys()):
            stats = width_stats[width]
            crossovers = [s["crossover"] for s in stats if s["crossover"] is not None]
            avg_cross = (
                sum(crossovers) / len(crossovers) if crossovers else ""
            )
            pct_cross = (
                100.0 * len(crossovers) / len(stats) if stats else 0.0
            )
            avg_eff = sum(s["efficiency_ratio"] for s in stats) / len(stats)
            avg_vedic_min = sum(s["vedic_min"] for s in stats) / len(stats)
            avg_school_min = sum(s["school_min"] for s in stats) / len(stats)
            avg_cp_ratio = (
                sum(
                    s["school_min"] / s["vedic_min"]
                    for s in stats
                    if s["vedic_min"]
                )
                / len(stats)
            )

            writer.writerow(
                {
                    "digit_width": width,
                    "avg_crossover_workers": round(avg_cross, 4)
                    if crossovers
                    else "",
                    "pct_pairs_with_crossover": round(pct_cross, 4),
                    "avg_efficiency_ratio": round(avg_eff, 6),
                    "vedic_avg_min_completion": round(avg_vedic_min, 4),
                    "school_avg_min_completion": round(avg_school_min, 4),
                    "avg_critical_path_ratio": round(avg_cp_ratio, 6),
                }
            )

    print(f"Wrote {output_path} ({len(pairs)} pairs)")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2B DAG simulator")
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=_REPO_ROOT / "pairs.json",
    )
    parser.add_argument(
        "--digit-widths",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "phase2b.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary CSV path (default: output dir / phase2b_summary.csv)",
    )
    parser.add_argument(
        "--skip-test-gate",
        action="store_true",
        help="Skip pytest gate (for development only)",
    )
    args = parser.parse_args()

    summary = args.summary
    if summary is None:
        summary = args.output.parent / "phase2b_summary.csv"

    run_benchmark(
        args.pairs_json,
        args.digit_widths,
        args.output,
        summary,
        skip_test_gate=args.skip_test_gate,
    )


if __name__ == "__main__":
    main()
