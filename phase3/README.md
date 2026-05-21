# Phase 3 — CUDA Kernels and DAG Prediction Validation

GPU implementation of instrumented Vedic (Urdhva-Tiryagbhyam) and schoolbook multiplication, with correctness validation and timing to compare against the Phase 2B DAG simulator.

## Target hardware

Designed and built for:

| Property | Value |
|----------|--------|
| GPU | NVIDIA RTX 3050 Ti Laptop |
| Compute capability | 8.6 (`CMAKE_CUDA_ARCHITECTURES=86`) |
| Warp size | 32 |

## Digit representation

Matches Phase 1 and Phase 2A exactly:

- Base-10, **LSB at index 0**
- `int` digit arrays on device; `long long` for final integer reconstruction
- `__device__ __constant__ int MULT_TABLE[10][10]`
- Operands padded to length `n` before every kernel launch
- Inter-column Vedic carry is a **digit list** (`carry_digits[1:]`), not a single scalar — see Phase 1 `vedic.py`

## Why Option A (sequential column launches)

Columns depend on the previous column’s carry vector. Cooperative multi-column kernels cannot preserve Phase 1 semantics without global synchronization. Option A launches **one column kernel per `k`**, with `cudaDeviceSynchronize()` between columns — correct, easy to validate, and aligned with Phase 2B’s per-column schedulable layers.

## Phase 2B workers ↔ Phase 3 threads

| Phase 2B | Phase 3 |
|----------|---------|
| `workers` (DAG simulator) | `threads_per_column_block` (Vedic column block size) |
| `workers = -1` (unlimited) | Not timed separately; critical path is separate analysis |

**Vedic:** `threads_per_column_block` limits how many threads share the `p` cross-products in one column. If `t < p`, threads handle multiple products serially; if `t ≥ p`, extra threads are idle.

**Schoolbook:** Pass 1 uses `row_threads` (= `threads_per_column_block` in CSV). Pass 2 is always 1×1 (sequential accumulation, matching Phase 2B DAG). Benchmark includes:

- `schoolbook` — fixed `row_threads = n` (natural row parallelism)
- `schoolbook_sweep` — same thread sweep as Vedic for comparison

## Absolute times will be slow at small n

Each column launch has kernel launch + sync overhead. At `n = 4–6`, warps are underutilized and sync dominates — **compare speedup ratios**, not raw µs vs Phase 2A CPU.

## Build requirements

- CMake 3.18+
- CUDA toolkit with `nvcc` (11.x+)
- C++17 / CUDA 17
- Internet for FetchContent (`nlohmann/json`)

## Build and run

From `phase3/`:

```powershell
# Visual Studio 2026 (VS 18) — adjust generator if you have VS 2022 instead
cmake -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```

**Validator runs automatically before `benchmark` links.** To run manually:

```powershell
.\build\validator.exe --input ..\pairs.json
```

All **160 pairs** (widths 2–9) must pass before timing.

Benchmark (widths 4–9 by default):

```powershell
.\build\benchmark.exe --input ..\pairs.json --digit-widths 4 5 6 7 8 9 --iterations 10000 --warmup 100 --output results\phase3.csv
```

Compare with Phase 2B:

```powershell
python analysis\compare_dag.py
```

## CSV columns (`phase3.csv`)

| Column | Meaning |
|--------|---------|
| `threads_per_column_block` | Active threads in Vedic column / schoolbook partial row |
| `mean_time_us`, `std_time_us` | GPU time per multiply (warmup excluded) |
| `speedup_vs_single_thread` | `mean(t=1) / mean(t)` |
| `method` | `vedic`, `schoolbook`, `schoolbook_sweep`, `native` |

## `prediction_accurate_within_2x`

From `compare_dag.py`: crossover prediction is accurate if  
`0.5 ≤ actual_crossover_threads / predicted_crossover_workers ≤ 2.0`.  
`model_valid` also requires Vedic max speedup (actual vs Phase 2B) within the same 2× band.

## Experiment outcomes

| Outcome | Condition |
|---------|-----------|
| **Validated** | ≥5/6 widths (4–9) have `model_valid=yes` |
| **Partial** | Speedup trends correlate but crossover or magnitude outside 2× |
| **Invalid** | GPU speedup flat or inverted vs Phase 2B — DAG not predictive on this GPU |

## Known limitation

Base-10 Urdhva at these widths has **strictly sequential columns** (carry vector chains). No inter-column GPU parallelism is possible without changing the algorithm. All parallelism is **within** each column.

## Layout

```
phase3/
├── kernels/          # .cuh device + host launchers
├── host/             # validator.cu, benchmark.cu
├── analysis/         # compare_dag.py
├── results/
├── CMakeLists.txt
└── README.md
```

## Scheduler note (Phase 2B)

Hand-built DAG test: `workers=4` yields **3** steps (not 4) because ADD nodes are sequential after parallel MULTs.
