# Vedic Mathematics Algorithm Benchmark

## Experiment goal

Measure how **Urdhva-Tiryagbhyam** (vertical-and-crosswise) multiplication compares to **schoolbook partial-products** multiplication and **native Python integers**, across operand sizes from 2 to 8 decimal digits.

The benchmark instruments every single-digit multiply, add, and carry step so operation counts are comparable across algorithms, then times each implementation with `timeit` and records theoretical parallelism metrics from a simplified DAG model.

## Hypothesis

1. **Correctness**: All three methods produce identical products for the same operands.
2. **Operation count**: Vedic and schoolbook both perform Θ(n²) single-digit multiplications for n-digit operands, but differ in how additions and carries are grouped (column-wise vs row-wise partial products).
3. **Theoretical parallelism**: Vedic column-wise diagonal products vs schoolbook row-wise partial products yield different `parallelism_score` profiles on the DAG model.
4. **Wall-clock time**: Native multiplication will dominate in CPython; instrumented Python algorithms are for **structural** comparison, not production speed.

## How to run

From the repository root (`vedic_math_ML`):

```bash
# Full benchmark (slow: 100k iterations × 7 widths × 20 pairs)
python -m vedic_benchmark.main

# Faster dev run
python -m vedic_benchmark.main --iterations 10000

# Quick smoke test with parallel workers
python -m vedic_benchmark.main --digits 2 --pairs 2 --iterations 1000 --repeat 3 --workers 2

# Custom widths, optional chart, skip CLI summary
python -m vedic_benchmark.main --digits 4 5 6 --pairs 10 --workers 4 --no-summary --chart
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--digits` | `2 3 4 5 6 7 8` | Operand digit widths to test |
| `--pairs` | `20` | Random (a, b) pairs per width |
| `--iterations` | `100000` | `timeit` loops per repeat sample |
| `--repeat` | `5` | Number of repeat samples (→ `time_repeat_1..N` columns) |
| `--workers` | `1` | Process pool size (`>1` uses multiprocessing, not threads) |
| `--run-id` | auto | Run identifier stored in every output row |
| `--no-summary` | off | Hide width-averaged CLI table (detail files still written) |
| `--chart` | off | Save `benchmark_chart.png` (requires matplotlib) |

## Output files

Results are flushed **after each operand pair** so an interrupted run keeps completed pairs.

### 1. Summary CSV — `vedic_benchmark/results/benchmark_results.csv`

One row per **(digit_width, pair_index, method)** with:

- `run_id`, `operand_a`, `operand_b`, `product` — exact calculation preserved
- `time_repeat_1` … `time_repeat_N` — per-call µs for each `timeit` repeat
- `mean_time_us`, `std_time_us` — derived convenience fields
- Operation counts and depth metrics (vedic/schoolbook only)

### 2. Detail JSONL — `vedic_benchmark/results/benchmark_detail.jsonl`

One JSON object per **(pair, method, repeat_index)** — maximum granularity for plotting and outlier analysis.

### CLI summary table

Unless `--no-summary` is set, a **width-averaged** table is printed to stdout. This aggregates across pairs for a quick overview; use the CSV/JSONL for per-calculation detail.

## Performance notes

- **Threads do not help**: Vedic and schoolbook are CPU-bound pure Python; the GIL limits `threading`.
- **`--workers N`**: Uses `multiprocessing` to time different (pair, method) jobs in parallel. Diminishing returns beyond CPU core count.
- **Publication vs dev**: `--iterations 100000` is the default for stable means; use `--iterations 10000` (or lower) while iterating.

## Metrics interpretation

| Metric | Meaning |
|--------|---------|
| `time_repeat_*` / `time_us` (JSONL) | Single repeat sample, microseconds per call |
| `mean_time_us` / `std_time_us` | Mean and stdev across repeats |
| `multiplications` | Single-digit × single-digit ops via lookup table |
| `additions` | Single-digit additions during accumulation |
| `carry_propagations` | Steps where a carry digit is produced and propagated |
| `total_ops` | Sum of the three counters above |
| `parallel_width` | Theoretical max parallel ops at one DAG level |
| `sequential_depth` | Theoretical dependent-chain length |
| `parallelism_score` | `parallel_width / sequential_depth` |

### Depth models

- **Vedic**: `parallel_width = N`; `sequential_depth = Σ_k (1 + ⌈log₂ p⌉ if p>1 + 1)` where `p = min(k+1, 2N−1−k)`.
- **Schoolbook**: `parallel_width = N`; `sequential_depth = N + (N−1)·N`.

## Phase 2 / Phase 3 notes

**Phase 2** — Port instrumented multiply to C++ for cleaner wall-clock timing.

**Phase 3** — Explore CUDA kernel scheduling using the DAG parallelism model.

## Requirements

- Python 3.10+
- Standard library only
- Optional: `matplotlib` for `--chart`
