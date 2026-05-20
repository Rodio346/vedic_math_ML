"""Benchmark runner: per-pair correctness, timing, CSV + JSONL export."""

from __future__ import annotations

import csv
import json
import logging
import random
import statistics
import sys
import timeit
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vedic_benchmark.algorithms import native as native_alg
from vedic_benchmark.algorithms import schoolbook as schoolbook_alg
from vedic_benchmark.algorithms import vedic as vedic_alg
from vedic_benchmark.analysis.depth import compare_depth

logger = logging.getLogger(__name__)

DEFAULT_DIGIT_WIDTHS = [2, 3, 4, 5, 6, 7, 8]
DEFAULT_PAIRS = 20
DEFAULT_ITERATIONS = 100_000
DEFAULT_REPEAT = 5

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
JSONL_PATH = RESULTS_DIR / "benchmark_detail.jsonl"

DEFAULT_METHODS = ("vedic", "schoolbook", "native")
METHODS = DEFAULT_METHODS  # backward compatibility

_CSV_STATIC_FIELDS = [
    "run_id",
    "digit_width",
    "pair_index",
    "operand_a",
    "operand_b",
    "product",
    "method",
]
_CSV_TAIL_FIELDS = [
    "mean_time_us",
    "std_time_us",
    "multiplications",
    "additions",
    "carry_propagations",
    "total_ops",
    "sequential_depth",
    "parallel_width",
    "parallelism_score",
]


def _csv_fieldnames(repeat: int) -> list[str]:
    repeat_cols = [f"time_repeat_{i + 1}" for i in range(repeat)]
    return _CSV_STATIC_FIELDS + repeat_cols + _CSV_TAIL_FIELDS


@dataclass
class BenchmarkRow:
    """One CSV row: one method on one operand pair."""

    run_id: str
    digit_width: int
    pair_index: int
    operand_a: int
    operand_b: int
    product: int
    method: str
    repeat_times_us: list[float] = field(default_factory=list)
    mean_time_us: float = 0.0
    std_time_us: float = 0.0
    multiplications: int = 0
    additions: int = 0
    carry_propagations: int = 0
    total_ops: int = 0
    sequential_depth: int = 0
    parallel_width: int = 0
    parallelism_score: float = 0.0


@dataclass
class WidthSummary:
    """Aggregated means for the CLI summary table (one row per digit width)."""

    digit_width: int
    vedic_time_us: float
    school_time_us: float
    native_time_us: float
    vedic_ops: float
    school_ops: float
    vedic_par: float
    school_par: float


@dataclass(frozen=True)
class BenchmarkJobArgs:
    """Picklable arguments for a single (pair, method) benchmark job."""

    run_id: str
    digit_width: int
    pair_index: int
    operand_a: int
    operand_b: int
    product: int
    method: str
    iterations: int
    repeat: int
    sequential_depth: int
    parallel_width: int
    parallelism_score: float


def _run_vedic(a: int, b: int) -> int:
    return vedic_alg.multiply(a, b)[0]


def _run_schoolbook(a: int, b: int) -> int:
    return schoolbook_alg.multiply(a, b)[0]


def _run_native(a: int, b: int) -> int:
    return native_alg.multiply(a, b)


_TIMERS = {
    "vedic": _run_vedic,
    "schoolbook": _run_schoolbook,
    "native": _run_native,
}


def _digit_range(n_digits: int) -> tuple[int, int]:
    low = 10 ** (n_digits - 1)
    high = 10**n_digits - 1
    return low, high


def _random_pairs(
    n_digits: int,
    count: int,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    low, high = _digit_range(n_digits)
    return [(rng.randint(low, high), rng.randint(low, high)) for _ in range(count)]


def _assert_correctness(a: int, b: int, digit_width: int, pair_index: int) -> int:
    vedic_result, _ = vedic_alg.multiply(a, b)
    school_result, _ = schoolbook_alg.multiply(a, b)
    native_result = native_alg.multiply(a, b)
    if not (vedic_result == school_result == native_result):
        msg = (
            f"CORRECTNESS FAILURE width={digit_width} pair={pair_index} "
            f"({a}, {b}): vedic={vedic_result}, schoolbook={school_result}, "
            f"native={native_result}"
        )
        logger.error(msg)
        print(msg, file=sys.stderr)
        raise SystemExit(1)
    return native_result


def _time_method_us(
    a: int,
    b: int,
    method: str,
    iterations: int,
    repeat: int,
) -> list[float]:
    fn = _TIMERS[method]
    timer = timeit.Timer(lambda: fn(a, b))
    samples = timer.repeat(repeat=repeat, number=iterations)
    return [(t / iterations) * 1_000_000 for t in samples]


def _op_counts(a: int, b: int, method: str) -> tuple[int, int, int, int]:
    if method == "vedic":
        _, ctr = vedic_alg.multiply(a, b)
    elif method == "schoolbook":
        _, ctr = schoolbook_alg.multiply(a, b)
    else:
        return 0, 0, 0, 0
    return (
        ctr.multiplications,
        ctr.additions,
        ctr.carry_propagations,
        ctr.total_ops,
    )


def _benchmark_job(args: BenchmarkJobArgs) -> BenchmarkRow:
    repeat_times = _time_method_us(
        args.operand_a,
        args.operand_b,
        args.method,
        args.iterations,
        args.repeat,
    )
    mults, adds, carries, total = _op_counts(
        args.operand_a,
        args.operand_b,
        args.method,
    )
    mean_us = statistics.mean(repeat_times)
    std_us = statistics.stdev(repeat_times) if len(repeat_times) > 1 else 0.0

    return BenchmarkRow(
        run_id=args.run_id,
        digit_width=args.digit_width,
        pair_index=args.pair_index,
        operand_a=args.operand_a,
        operand_b=args.operand_b,
        product=args.product,
        method=args.method,
        repeat_times_us=repeat_times,
        mean_time_us=mean_us,
        std_time_us=std_us,
        multiplications=mults,
        additions=adds,
        carry_propagations=carries,
        total_ops=total,
        sequential_depth=args.sequential_depth,
        parallel_width=args.parallel_width,
        parallelism_score=args.parallelism_score,
    )


def _build_row(
    run_id: str,
    digit_width: int,
    pair_index: int,
    method: str,
    a: int,
    b: int,
    product: int,
    iterations: int,
    repeat: int,
    depth_map: dict,
) -> BenchmarkRow:
    seq_depth = 0
    par_width = 0
    par_score = 0.0
    if method in depth_map:
        dm = depth_map[method]
        seq_depth = dm.sequential_depth
        par_width = dm.parallel_width
        par_score = dm.parallelism_score

    job = BenchmarkJobArgs(
        run_id=run_id,
        digit_width=digit_width,
        pair_index=pair_index,
        operand_a=a,
        operand_b=b,
        product=product,
        method=method,
        iterations=iterations,
        repeat=repeat,
        sequential_depth=seq_depth,
        parallel_width=par_width,
        parallelism_score=par_score,
    )
    return _benchmark_job(job)


def _row_to_csv_dict(row: BenchmarkRow, repeat: int) -> dict[str, object]:
    data: dict[str, object] = {
        "run_id": row.run_id,
        "digit_width": row.digit_width,
        "pair_index": row.pair_index,
        "operand_a": row.operand_a,
        "operand_b": row.operand_b,
        "product": row.product,
        "method": row.method,
        "mean_time_us": f"{row.mean_time_us:.3f}",
        "std_time_us": f"{row.std_time_us:.3f}",
        "multiplications": row.multiplications,
        "additions": row.additions,
        "carry_propagations": row.carry_propagations,
        "total_ops": row.total_ops,
        "sequential_depth": row.sequential_depth,
        "parallel_width": row.parallel_width,
        "parallelism_score": row.parallelism_score,
    }
    for i in range(repeat):
        key = f"time_repeat_{i + 1}"
        data[key] = (
            f"{row.repeat_times_us[i]:.3f}"
            if i < len(row.repeat_times_us)
            else ""
        )
    return data


def _init_output_files(
    csv_path: Path,
    jsonl_path: Path,
    repeat: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(repeat)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
    jsonl_path.write_text("", encoding="utf-8")


def _append_csv_rows(
    rows: list[BenchmarkRow],
    path: Path,
    repeat: int,
) -> None:
    fieldnames = _csv_fieldnames(repeat)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(_row_to_csv_dict(row, repeat))


def _append_jsonl_rows(rows: list[BenchmarkRow], path: Path) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            for repeat_index, time_us in enumerate(row.repeat_times_us):
                record = {
                    "run_id": row.run_id,
                    "digit_width": row.digit_width,
                    "pair_index": row.pair_index,
                    "operand_a": row.operand_a,
                    "operand_b": row.operand_b,
                    "product": row.product,
                    "method": row.method,
                    "repeat_index": repeat_index,
                    "time_us": round(time_us, 3),
                    "multiplications": row.multiplications,
                    "additions": row.additions,
                    "carry_propagations": row.carry_propagations,
                    "total_ops": row.total_ops,
                    "sequential_depth": row.sequential_depth,
                    "parallel_width": row.parallel_width,
                    "parallelism_score": row.parallelism_score,
                }
                fh.write(json.dumps(record) + "\n")


def _method_index(method: str, method_order: tuple[str, ...]) -> int:
    return method_order.index(method)


def _flush_pair_results(
    rows: list[BenchmarkRow],
    csv_path: Path,
    jsonl_path: Path,
    repeat: int,
    method_order: tuple[str, ...] = DEFAULT_METHODS,
) -> None:
    ordered = sorted(rows, key=lambda r: _method_index(r.method, method_order))
    _append_csv_rows(ordered, csv_path, repeat)
    _append_jsonl_rows(ordered, jsonl_path)


def _benchmark_pair_sequential(
    run_id: str,
    width: int,
    pair_index: int,
    a: int,
    b: int,
    product: int,
    iterations: int,
    repeat: int,
    depth: dict,
    method_order: tuple[str, ...] = DEFAULT_METHODS,
) -> list[BenchmarkRow]:
    return [
        _build_row(run_id, width, pair_index, method, a, b, product, iterations, repeat, depth)
        for method in method_order
    ]


def _benchmark_pair_parallel(
    run_id: str,
    width: int,
    pair_index: int,
    a: int,
    b: int,
    product: int,
    iterations: int,
    repeat: int,
    depth: dict,
    executor: ProcessPoolExecutor,
    method_order: tuple[str, ...] = DEFAULT_METHODS,
) -> list[BenchmarkRow]:
    futures = {}
    for method in method_order:
        seq_depth = 0
        par_width = 0
        par_score = 0.0
        if method in depth:
            dm = depth[method]
            seq_depth = dm.sequential_depth
            par_width = dm.parallel_width
            par_score = dm.parallelism_score

        job = BenchmarkJobArgs(
            run_id=run_id,
            digit_width=width,
            pair_index=pair_index,
            operand_a=a,
            operand_b=b,
            product=product,
            method=method,
            iterations=iterations,
            repeat=repeat,
            sequential_depth=seq_depth,
            parallel_width=par_width,
            parallelism_score=par_score,
        )
        futures[executor.submit(_benchmark_job, job)] = method

    rows_by_method: dict[str, BenchmarkRow] = {}
    for future in as_completed(futures):
        row = future.result()
        rows_by_method[row.method] = row

    return [rows_by_method[method] for method in method_order]


def _aggregate_summaries(rows: list[BenchmarkRow]) -> list[WidthSummary]:
    widths = sorted({r.digit_width for r in rows})
    summaries: list[WidthSummary] = []

    for width in widths:
        width_rows = [r for r in rows if r.digit_width == width]

        def mean_for(method: str, attr: str) -> float:
            subset = [r for r in width_rows if r.method == method]
            if not subset:
                return 0.0
            if attr == "mean_time_us":
                return statistics.mean(r.mean_time_us for r in subset)
            if attr == "total_ops":
                return statistics.mean(r.total_ops for r in subset)
            if attr == "parallelism_score":
                vals = [r.parallelism_score for r in subset if r.parallelism_score]
                return statistics.mean(vals) if vals else 0.0
            return 0.0

        summaries.append(
            WidthSummary(
                digit_width=width,
                vedic_time_us=mean_for("vedic", "mean_time_us"),
                school_time_us=mean_for("schoolbook", "mean_time_us"),
                native_time_us=mean_for("native", "mean_time_us"),
                vedic_ops=mean_for("vedic", "total_ops"),
                school_ops=mean_for("schoolbook", "total_ops"),
                vedic_par=mean_for("vedic", "parallelism_score"),
                school_par=mean_for("schoolbook", "parallelism_score"),
            )
        )
    return summaries


def _make_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def run_benchmark(
    digit_widths: list[int] | None = None,
    pairs_per_width: int = DEFAULT_PAIRS,
    iterations: int = DEFAULT_ITERATIONS,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 42,
    run_id: str | None = None,
    workers: int = 1,
    csv_path: Path | None = None,
    jsonl_path: Path | None = None,
    method_order: tuple[str, ...] | None = None,
) -> tuple[list[BenchmarkRow], list[WidthSummary], str]:
    """Run benchmark; flush CSV + JSONL after each pair; return rows and summaries."""
    widths = digit_widths or DEFAULT_DIGIT_WIDTHS
    out_csv = csv_path or CSV_PATH
    out_jsonl = jsonl_path or JSONL_PATH
    resolved_run_id = _make_run_id(run_id)
    order = method_order or DEFAULT_METHODS
    for name in order:
        if name not in DEFAULT_METHODS:
            raise ValueError(f"unknown method in method_order: {name}")

    _init_output_files(out_csv, out_jsonl, repeat)
    all_rows: list[BenchmarkRow] = []

    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)

    try:
        for width in widths:
            pairs = _random_pairs(width, pairs_per_width, seed=seed + width)
            depth = compare_depth(width)

            for pair_index, (a, b) in enumerate(pairs):
                product = _assert_correctness(a, b, width, pair_index)

                if pool is None:
                    pair_rows = _benchmark_pair_sequential(
                        resolved_run_id,
                        width,
                        pair_index,
                        a,
                        b,
                        product,
                        iterations,
                        repeat,
                        depth,
                        order,
                    )
                else:
                    pair_rows = _benchmark_pair_parallel(
                        resolved_run_id,
                        width,
                        pair_index,
                        a,
                        b,
                        product,
                        iterations,
                        repeat,
                        depth,
                        pool,
                        order,
                    )

                _flush_pair_results(pair_rows, out_csv, out_jsonl, repeat, order)
                all_rows.extend(pair_rows)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return all_rows, _aggregate_summaries(all_rows), resolved_run_id


def _load_pairs_from_json(pairs_file: Path) -> list[tuple[int, int, int, int]]:
    """Return list of (digit_width, pair_index, operand_a, operand_b)."""
    import json

    data = json.loads(pairs_file.read_text(encoding="utf-8"))
    result: list[tuple[int, int, int, int]] = []
    for entry in data["pairs"]:
        result.append(
            (
                int(entry["digit_width"]),
                int(entry["pair_index"]),
                int(entry["operand_a"]),
                int(entry["operand_b"]),
            )
        )
    return result


def run_benchmark_from_pairs(
    pairs_file: Path,
    iterations: int = DEFAULT_ITERATIONS,
    repeat: int = DEFAULT_REPEAT,
    run_id: str | None = None,
    workers: int = 1,
    csv_path: Path | None = None,
    jsonl_path: Path | None = None,
    method_order: tuple[str, ...] | None = None,
) -> tuple[list[BenchmarkRow], list[WidthSummary], str]:
    """Run benchmark using operand pairs from pairs.json (Phase 2A ground truth)."""
    out_csv = csv_path or CSV_PATH
    out_jsonl = jsonl_path or JSONL_PATH
    resolved_run_id = _make_run_id(run_id)
    order = method_order or DEFAULT_METHODS
    for name in order:
        if name not in DEFAULT_METHODS:
            raise ValueError(f"unknown method in method_order: {name}")

    _init_output_files(out_csv, out_jsonl, repeat)
    all_rows: list[BenchmarkRow] = []

    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)

    try:
        for width, pair_index, a, b in _load_pairs_from_json(pairs_file):
            depth = compare_depth(width)
            product = _assert_correctness(a, b, width, pair_index)

            if pool is None:
                pair_rows = _benchmark_pair_sequential(
                    resolved_run_id,
                    width,
                    pair_index,
                    a,
                    b,
                    product,
                    iterations,
                    repeat,
                    depth,
                    order,
                )
            else:
                pair_rows = _benchmark_pair_parallel(
                    resolved_run_id,
                    width,
                    pair_index,
                    a,
                    b,
                    product,
                    iterations,
                    repeat,
                    depth,
                    pool,
                    order,
                )

            _flush_pair_results(pair_rows, out_csv, out_jsonl, repeat, order)
            all_rows.extend(pair_rows)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return all_rows, _aggregate_summaries(all_rows), resolved_run_id
