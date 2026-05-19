"""CLI entry point for the Vedic Mathematics Algorithm Benchmark."""

from __future__ import annotations

import argparse
import sys

from vedic_benchmark.benchmark.runner import (
    CSV_PATH,
    DEFAULT_DIGIT_WIDTHS,
    DEFAULT_ITERATIONS,
    DEFAULT_PAIRS,
    DEFAULT_REPEAT,
    JSONL_PATH,
    run_benchmark,
)


def _print_summary_table(summaries) -> None:
    print("=== Width averages (optional summary) ===")
    header = (
        f"{'Width':>5}  {'vedic_us':>10}  {'school_us':>10}  {'native_us':>10}  "
        f"{'vedic_ops':>10}  {'school_ops':>10}  {'vedic_par':>10}  {'school_par':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        print(
            f"{row.digit_width:>5}  "
            f"{row.vedic_time_us:>10.2f}  {row.school_time_us:>10.2f}  "
            f"{row.native_time_us:>10.2f}  "
            f"{row.vedic_ops:>10.1f}  {row.school_ops:>10.1f}  "
            f"{row.vedic_par:>10.4f}  {row.school_par:>10.4f}"
        )


def _maybe_chart(summaries, digit_widths: list[int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.", file=sys.stderr)
        return

    widths = sorted(set(digit_widths))
    fig, ax = plt.subplots()
    ax.plot(
        widths,
        [s.vedic_time_us for s in summaries],
        marker="o",
        label="vedic",
    )
    ax.plot(
        widths,
        [s.school_time_us for s in summaries],
        marker="s",
        label="schoolbook",
    )
    ax.plot(
        widths,
        [s.native_time_us for s in summaries],
        marker="^",
        label="native",
    )
    ax.set_xlabel("Digit width")
    ax.set_ylabel("Mean time per multiply (µs)")
    ax.set_title("Multiplication algorithm benchmark")
    ax.legend()
    ax.set_xticks(widths)
    fig.tight_layout()
    chart_path = CSV_PATH.parent / "benchmark_chart.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Vedic vs schoolbook vs native multiplication.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        default=DEFAULT_DIGIT_WIDTHS,
        help="Digit widths to test (default: all 2-8).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"timeit iterations per repeat (default: {DEFAULT_ITERATIONS}).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=f"timeit repeat count per sample (default: {DEFAULT_REPEAT}).",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help=f"Random operand pairs per width (default: {DEFAULT_PAIRS}).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Benchmark run identifier (auto-generated if omitted).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process pool size for parallel timing (default: 1 = sequential).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the width-averaged CLI summary table.",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Plot mean times with matplotlib (optional dependency).",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 1
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 1

    run_label = args.run_id if args.run_id else "(auto)"
    print(
        f"Running benchmark: run_id={run_label}, widths={args.digits}, pairs={args.pairs}, "
        f"iterations={args.iterations}, repeat={args.repeat}, workers={args.workers}"
    )
    _rows, summaries, run_id = run_benchmark(
        digit_widths=args.digits,
        pairs_per_width=args.pairs,
        iterations=args.iterations,
        repeat=args.repeat,
        run_id=args.run_id,
        workers=args.workers,
    )
    print(f"\nRun ID: {run_id}")
    print(f"Summary CSV: {CSV_PATH}")
    print(f"Detail JSONL: {JSONL_PATH}\n")

    if not args.no_summary:
        _print_summary_table(summaries)

    if args.chart:
        _maybe_chart(summaries, args.digits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
