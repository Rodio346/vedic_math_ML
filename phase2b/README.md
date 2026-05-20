# Phase 2B — DAG Simulator and Crossover Analysis

Phase 2B models **Urdhva-Tiryagbhyam** (Vedic) and **schoolbook** multiplication as directed acyclic graphs of single-digit operations, simulates parallel execution under a fixed worker pool, and finds the worker count at which Vedic’s completion time meets or beats schoolbook’s.

Unlike Phase 1 [`depth.py`](../vedic_benchmark/analysis/depth.py), which uses **worst-case** carry layers per column width, Phase 2B DAGs are built from **actual operand values**: CARRY nodes appear only when real arithmetic produces a carry.

## What this tests and why

| Phase | Question |
|-------|----------|
| **Phase 1** | How much work (ops) and theoretical depth per digit width? |
| **Phase 2A** | Are counts and timing reproducible in C++ on shared `pairs.json`? |
| **Phase 2B** | Given real dependency graphs, how does **simulated parallel completion time** scale with workers, and where does Vedic cross over schoolbook? |

This bridges operation counts to **scheduling**: if Vedic’s critical path is shorter but schoolbook wins at low worker counts, hardware parallelism matters.

## Connection to Phase 2A

- Uses the same [`pairs.json`](../pairs.json) (160 pairs, widths 2–9).
- DAG node counts match Phase 1 `OperationCounter` totals (MULT + ADD + CARRY = `total_ops`).
- Phase 2A CSV compares **wall-clock**; Phase 2B compares **discrete-time DAG schedules**.

## Requirements

- Python 3.10+
- `pytest` (see repo root `requirements-dev.txt`)
- Repo root on `PYTHONPATH` (handled by `conftest.py` when running tests from `phase2b/`)

## Run tests first (required)

From `phase2b/`:

```powershell
cd phase2b
pip install -r ..\requirements-dev.txt
python -m pytest tests/ -v
```

`main.py` refuses to run until `pytest tests/` passes.

## Run benchmark

From `phase2b/`:

```powershell
python main.py --pairs-json ..\pairs.json --digit-widths 2 3 4 5 6 7 8 9 --output results\phase2b.csv
```

Outputs:

- `results/phase2b.csv` — one row per `(pair, algorithm, worker_count)`
- `results/phase2b_summary.csv` — one row per `digit_width`

### CSV columns (`phase2b.csv`)

| Column | Meaning |
|--------|---------|
| `workers` | Worker pool size; **`-1` means unlimited** parallel workers (critical path / `theoretical_min`) |
| `completion_time` | Simulated steps to finish all nodes |
| `theoretical_min` | Critical path length (`workers=-1` row) |
| `utilisation` | `total_ops / (workers × completion_time)` |
| `speedup_vs_serial` | `completion_time(workers=1) / completion_time` |
| `crossover_workers` | Minimum W>1 where Vedic completion **<** schoolbook (empty if none) |
| `efficiency_ratio` | See definition below |

`efficiency_ratio = schoolbook_min_completion_steps / vedic_min_completion_steps`. Values greater than 1.0 mean Vedic has a shorter critical path and is more hardware-efficient at unlimited parallelism. Values below 1.0 mean schoolbook has a shorter critical path for that specific operand pair.

Worker sweep per pair: `1, 2, 4, 8, n, n//2+1, 2n, n², unlimited`, plus `n+1, n+2, n+4` when `n ≥ 7`, where `n` is padded operand width.

## Interpreting results

### `crossover_workers`

Minimum worker count **W > 1** in the tested set where **Vedic simulated completion is strictly less than schoolbook**. `workers=1` is excluded (both run serially; ties are not informative). Empty means no crossover in the sweep (see outcomes A/B/C below).

### `efficiency_ratio`

See the column definition above (`schoolbook_min / vedic_min` at unlimited workers).

### Research outcomes

| Outcome | Condition | Meaning |
|---------|-----------|---------|
| **A** | `crossover_workers = W` (finite) | Vedic wins only when parallelism ≥ W; threshold for hardware width. |
| **B** | No crossover; schoolbook faster at all tested W | Schoolbook’s schedule dominates; extra Vedic parallelism does not overcome longer paths/more ops in this model. |
| **C** | No crossover; Vedic < schoolbook at all tested W>1 | Vedic wins on parallel structure without a finite crossover threshold. |

## Layout

```
phase2b/
├── dag/
│   ├── node.py
│   ├── vedic_dag.py
│   ├── schoolbook_dag.py
│   └── scheduler.py
├── analysis/
│   └── crossover.py
├── tests/
│   ├── test_dag_correctness.py
│   └── test_scheduler.py
├── results/
├── main.py
└── README.md
```

## Scheduler note

The hand-built scheduler test DAG uses 4 parallel MULTs and chained ADDs. With `workers=4`, completion time is **3** steps (not 4): all MULTs in step 1, then sequential ADDs. The simulator follows dependency edges, not an optimistic upper bound.

## Operand-specific carries

Example: `10×10` produces fewer CARRY nodes than `99×99` at the same padded width. Phase 1 depth metrics for width `N` still use worst-case column overflow; Phase 2B does not — compare `avg_critical_path_ratio` in the summary CSV to Phase 1 `sequential_depth` trends for a structural vs worst-case contrast.
