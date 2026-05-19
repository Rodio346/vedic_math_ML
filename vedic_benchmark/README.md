# `vedic_benchmark` package

This directory contains **Phase 1** of the Vedic Math ML project.

**Start with the main project README:** [`../README.md`](../README.md) — overview, quick start, output format, and roadmap.

## Run the benchmark

From the repository root:

```bash
python -m vedic_benchmark.main
```

## Package map

| Path | Role |
|------|------|
| `main.py` | CLI (`--digits`, `--pairs`, `--iterations`, `--workers`, …) |
| `algorithms/vedic.py` | Urdhva-Tiryagbhyam multiply |
| `algorithms/schoolbook.py` | Schoolbook partial products |
| `algorithms/native.py` | `int` multiply baseline |
| `algorithms/_digits.py` | Digit conversion + instrumented single-digit ops |
| `analysis/counter.py` | `OperationCounter` |
| `analysis/depth.py` | Theoretical `parallel_width`, `sequential_depth`, `parallelism_score` |
| `benchmark/runner.py` | Correctness gate, `timeit`, CSV + JSONL export |
| `results/` | Generated `benchmark_results.csv` and `benchmark_detail.jsonl` (gitignored) |
