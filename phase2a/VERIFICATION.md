# Phase 2A verification

## Counter parity (Python ↔ C++)

Golden operation counts live in `verification/counter_parity.json` (20 entries: 5 pairs × widths 2 and 3 × vedic + schoolbook). They were generated from Phase 1 Python via `tools/build_counter_parity.py`.

The `verify` executable loads that file and runs instrumented `vedic::multiply` / `schoolbook::multiply`. **All 20 must pass** before `benchmark` builds (`run_tests` target).

```powershell
cmake --build build --target run_tests
```

Expected tail:

```
Summary: 20/20 passed
```

## Unit tests

- `test_digits` — digit conversion and instrumented single-digit ops
- `test_algorithms` — product correctness vs native `a*b`, vedic mult count = N² for non-zero operands, schoolbook mult count = len(a)×len(b)

## Shared pairs file

Phase 1 and Phase 2A must use the same `pairs.json`:

| Field | Value |
|-------|--------|
| Path | `pairs.json` (repo root) |
| Pairs | 160 (20 per digit width 2–9) |
| Seed | 42 |
| SHA256 | `2906ca08a5982cd30b77c390bbcb11037399554301984194466993d2c920088b` |

Regenerate:

```powershell
python phase2a/tools/export_pairs.py
```

## Timing methodology

Matches Phase 1 intent:

1. One **instrumented** multiply per pair/method for operation counts.
2. **Warmup** then timed loops using `multiply_fast` (no counter overhead).
3. `mean_time_us` / `std_time_us` from `--repeats` samples.

CSV columns align with Phase 1 plus `compiler_flags`, `warmup_iterations`, `platform`.

## Cross-phase comparison (`compare_phases.py`)

Per digit width `w`, the tool aggregates mean `mean_time_us` across pairs:

| Symbol | Definition |
|--------|------------|
| `vs_py` | `school_py / vedic_py` |
| `vs_cpp` | `school_cpp / vedic_cpp` |
| `d_vs` (`ratio_change`) | `vs_cpp - vs_py` |

`py/cpp_v` and `py/cpp_s` are C++ mean divided by Python mean for each method (absolute speedup factor between languages, not algorithm comparison).

**Interpretation:** If `d_vs` is near zero for a width, the *relative* schoolbook vs vedic advantage is similar in C++ and Python despite different absolute times.

## Manual spot checks

| Pair | Method | mults | adds | carries |
|------|--------|-------|------|---------|
| 23×41 | vedic | 4 | 6 | 1 |
| 23×41 | schoolbook | 4 | 5 | 1 |
| 62×76 | vedic | 4 | 12 | 4 |
| 62×76 | schoolbook | 4 | 9 | 5 |

Depth (from `digit_width` in pairs, not recomputed operand width):

- Vedic: `sequential_depth = 2N - 1`, `parallel_width = N`
- Schoolbook: `sequential_depth = 2N - 1`, `parallel_width = N²`

## Build flags

Release build uses **`-O2` only** (no SIMD intrinsics). CMake sets `-std=c++17` via `CMAKE_CXX_STANDARD`.
