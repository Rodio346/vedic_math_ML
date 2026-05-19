# Vedic Math ML — Algorithm Benchmark (Phase 1)

Research project asking whether **Vedic multiplication** (Urdhva-Tiryagbhyam) exposes more **parallel structure** than grade-school multiplication — properties that matter on GPUs and custom hardware, not whether Python can beat the built-in `*` operator.

Phase 1 is a **Python benchmark** that measures, for each random operand pair:

1. **Wall-clock time** (microseconds per multiply)
2. **Raw operation counts** (every single-digit multiply, add, and carry)
3. **Theoretical parallelism** (dependency-graph width, depth, and score)

Results are saved per calculation so you can analyze variance, outliers, and scaling from 2- through 8-digit operands.

---

## Why this exists

| Question | What we compare |
|----------|-----------------|
| Is the math correct? | Vedic vs schoolbook vs `int` multiply — must match exactly |
| How much “work” does each algorithm do? | Instrumented single-digit operation counters |
| How parallel is the *structure*? | DAG model: `parallelism_score = parallel_width / sequential_depth` |
| How fast is it in CPython? | `timeit` timing (dominated by interpreter overhead for instrumented code) |

**Hypothesis:** Urdhva-Tiryagbhyam groups work **by output column** (all products with `i + j = k` before carry), while schoolbook builds **partial-product rows**. That difference may show up in parallelism scores even when Python’s native `int` multiply is fastest overall.

**Not in scope for Phase 1:** NumPy, CUDA, or beating production libraries. Those are planned for later phases.

---

## Repository layout

```
vedic_math_ML/
├── README.md                 ← you are here
├── .gitignore
└── vedic_benchmark/          ← Phase 1 package
    ├── main.py               ← CLI entry point
    ├── algorithms/
    │   ├── vedic.py          ← Urdhva-Tiryagbhyam (column-wise)
    │   ├── schoolbook.py     ← Partial-products method
    │   ├── native.py         ← Python int * (timing baseline)
    │   └── _digits.py        ← Digit lists + instrumented primitives
    ├── analysis/
    │   ├── counter.py        ← OperationCounter
    │   └── depth.py          ← Theoretical DAG metrics
    ├── benchmark/
    │   └── runner.py           ← Correctness, timing, CSV + JSONL export
    └── results/              ← Generated at runtime (gitignored)
        ├── benchmark_results.csv
        └── benchmark_detail.jsonl
```

---

## Requirements

- **Python 3.10+**
- **Standard library only** for the benchmark core
- **Optional:** `matplotlib` if you use `--chart`

No `pip install` is required for the benchmark itself. Clone the repo and run from the project root.

---

## Quick start

Open a terminal in the repository root (`vedic_math_ML`):

```powershell
# Smoke test (~10 seconds)
python -m vedic_benchmark.main --digits 2 --pairs 2 --iterations 1000 --repeat 3

# Faster development run
python -m vedic_benchmark.main --iterations 10000 --workers 4

# Full experiment (slow: 7 widths × 20 pairs × 100k iterations)
python -m vedic_benchmark.main --workers 4
```

After a run you will see:

- **Run ID** in the console
- **`vedic_benchmark/results/benchmark_results.csv`** — one row per (pair, method)
- **`vedic_benchmark/results/benchmark_detail.jsonl`** — one line per timing repeat

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--digits` | `2 3 4 5 6 7 8` | Operand digit widths (each width uses numbers with exactly that many digits) |
| `--pairs` | `20` | Random `(a, b)` pairs generated per width |
| `--iterations` | `100000` | Inner `timeit` loop count per repeat sample |
| `--repeat` | `5` | Number of repeat samples → `time_repeat_1` … `time_repeat_N` in CSV |
| `--workers` | `1` | `>1` runs timing jobs in **separate processes** (not threads) |
| `--run-id` | auto | Identifier stored in every output row |
| `--no-summary` | off | Skip the width-averaged table on stdout |
| `--chart` | off | Write `benchmark_chart.png` (needs matplotlib) |

**Examples**

```powershell
# Only 4-digit operands, 10 pairs
python -m vedic_benchmark.main --digits 4 --pairs 10

# Digit widths 2 through 5 only
python -m vedic_benchmark.main --digits 2 3 4 5

# Parallel timing on 4 CPU cores
python -m vedic_benchmark.main --workers 4 --iterations 10000
```

---

## What each algorithm does

### Urdhva-Tiryagbhyam (`vedic.py`)

For two `N`-digit operands (base 10, LSB-first digit lists):

1. For each output column index `k` from `0` to `2N - 2`, compute all products `a[i] * b[j]` where `i + j = k`.
2. Sum those products into a column total using instrumented adds.
3. Write the units digit to the result and carry the rest into column `k + 1`.

Every single-digit multiply and add goes through `OperationCounter`.

### Schoolbook (`schoolbook.py`)

Classic partial products:

1. For each digit of the multiplier, multiply the entire multiplicand digit-by-digit.
2. Shift the partial row by the digit position.
3. Add into an accumulator with explicit carries.

Same instrumentation rules as Vedic.

### Native (`native.py`)

Calls `a * b` with no counter. Used for **correctness checks** and a **wall-clock ceiling** (optimized CPython big integers).

---

## Output files explained

Results are **appended after each operand pair** (all three methods). If a long run is interrupted, completed pairs remain on disk.

### Summary CSV — `benchmark_results.csv`

One row per **`(digit_width, pair_index, method)`**.

| Column | Meaning |
|--------|---------|
| `run_id` | Benchmark invocation id |
| `operand_a`, `operand_b`, `product` | Exact integers for this row |
| `time_repeat_1` … `N` | Microseconds per call for each `timeit` repeat |
| `mean_time_us`, `std_time_us` | Mean and standard deviation across repeats |
| `multiplications`, `additions`, `carry_propagations` | Counted primitive ops (0 for native) |
| `total_ops` | Sum of the three counters |
| `sequential_depth`, `parallel_width`, `parallelism_score` | Theoretical DAG metrics (vedic/schoolbook) |

### Detail JSONL — `benchmark_detail.jsonl`

One JSON object per **`(pair, method, repeat_index)`** — finest granularity for plotting and filtering, e.g. in pandas:

```python
import pandas as pd
df = pd.read_json("vedic_benchmark/results/benchmark_detail.jsonl", lines=True)
df[df["method"] == "vedic"].groupby("digit_width")["time_us"].mean()
```

### CLI summary table

Unless you pass `--no-summary`, the program prints **one row per digit width** averaged over all pairs. That is only a quick overview; **use the CSV/JSONL for per-calculation analysis.**

---

## Metrics cheat sheet

| Metric | Interpretation |
|--------|----------------|
| **Timing** | Python overhead is large for instrumented code; compare *relative* trends across methods and widths, not absolute speed vs NumPy |
| **total_ops** | Structural “work” — comparable between Vedic and schoolbook |
| **parallelism_score** | Higher ⇒ more theoretical concurrency in the DAG model (not measured GPU occupancy) |

**Depth models** (from digit width `N`, see [`depth.py`](vedic_benchmark/analysis/depth.py)):

- **Vedic:** `parallel_width = N`; `sequential_depth = Σ_k (1 + ⌈log₂ p⌉ if p>1 + carry_layer)` where `p = min(k+1, 2N−1−k)`.
- **Schoolbook:** `parallel_width = N²` (all digit multiplies independent); `sequential_depth = 2N − 1` (row chain + sequential row accumulation).

---

## Verification (before Phase 2)

Four layers validate algorithms, counters, depth theory, and timing. Install dev deps and run from the repo root:

```powershell
pip install -r requirements-dev.txt
python -m pytest tests/ -v
python scripts/verify_counters.py
python scripts/audit_results.py
python scripts/benchmark_order_check.py
python scripts/cross_verify_sample.py --count 5
```

| Layer | What | Artifact |
|-------|------|----------|
| 1 | Product correctness (edge + mixed-width operands) | [`tests/test_correctness.py`](tests/test_correctness.py) |
| 2 | Operation counter golden values | [`scripts/verify_counters.py`](scripts/verify_counters.py), [`tests/test_counter_golden.py`](tests/test_counter_golden.py) |
| 3 | Depth formulas + CSV depth columns | [`tests/test_depth.py`](tests/test_depth.py), [`scripts/audit_results.py`](scripts/audit_results.py) |
| 4 | Timing variance / order bias | [`scripts/audit_results.py`](scripts/audit_results.py) (WARN lines), [`scripts/benchmark_order_check.py`](scripts/benchmark_order_check.py) |

**Instrumentation notes (Layer 2):**

- **Vedic** counts every `add` / `carry` via [`_digits.py`](vedic_benchmark/algorithms/_digits.py) plus explicit `carry()` when column overflow digits are promoted — this is the **implementation-defined** metric, not a minimal mathematical carry count.
- **Schoolbook** uses partial instrumentation in the partial-product row (`product + carry` with selective counter calls); treat schoolbook op totals as comparable trends, not identical accounting to Vedic.

**Iteration sensitivity (Layer 4, manual):** Run the same width with `--iterations 10000` vs `1000000` and compare `mean_time_us` in the CSV; means should agree within a few percent.

**Exit criteria for Phase 2:** all pytest tests green; `audit_results.py` reports 0 errors on your latest CSV; timing WARN lines reviewed or re-run with `--workers 1`.

---

## Correctness

Before timing each pair, the runner asserts:

```text
vedic(a, b) == schoolbook(a, b) == native(a, b)
```

On failure it logs the operands and exits immediately.

---

## Performance tips

| Tip | Reason |
|-----|--------|
| Use `--workers 4` (or your CPU core count) | Timing jobs are CPU-bound; **multiprocessing** helps, **threading** does not (GIL) |
| Lower `--iterations` while developing | Default `100000` is for stable means; `1000`–`10000` is enough for smoke tests |
| Benchmark outputs are gitignored | Regenerate locally; commit only source code |

---

## Roadmap

| Phase | Goal |
|-------|------|
| **1 (current)** | Operation counts, DAG parallelism, Python timing, granular CSV/JSONL |
| **2** | Port instrumented multiply to C++ for cleaner wall-clock measurement |
| **3** | CUDA / kernel scheduling informed by the DAG model |

---

## Further reading

Module-level docstrings in each `.py` file describe that file’s role in the experiment.

For a shorter duplicate of the CLI and metrics sections, see [`vedic_benchmark/README.md`](vedic_benchmark/README.md).

---

## Development

```powershell
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

---

## License

Add a license file if you plan to open-source this repository.
