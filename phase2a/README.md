# Phase 2A — C++ benchmark port

C++17 port of the instrumented Vedic (Urdhva-Tiryagbhyam) and schoolbook multipliers from Phase 1. Produces a CSV comparable to `vedic_benchmark/results/benchmark_results.csv` for cross-language timing and operation-count checks.

## Requirements

| Tool | Required? | Notes |
|------|-----------|--------|
| **CMake** 3.14+ | Yes | Configure and build |
| **g++** (MinGW) | Yes | Compiler for this project on Windows |
| **cl** (MSVC) | No | Optional; CMake uses MSVC flags if you choose a Visual Studio generator |

On Windows, **WinLibs MinGW** (`g++`, `mingw32-make`) is enough. You do **not** need Visual Studio or `cl` unless you prefer that toolchain.

### PATH (Windows)

Add these to your user **PATH** (adjust WinLibs path if winget installed a different version):

```
C:\Program Files\CMake\bin
C:\Users\<you>\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin
```

Install if missing:

```powershell
winget install Kitware.CMake
winget install BrechtSanders.WinLibs.POSIX.UCRT
```

## Build

From the repo root:

```powershell
cd phase2a
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
cmake --build build
```

`benchmark` depends on `run_tests`: unit tests and 20/20 counter parity against Python must pass before the benchmark binary is linked.

Run tests only:

```powershell
cmake --build build --target run_tests
```

## Benchmark

Uses shared `pairs.json` at the repo root (160 pairs, digit widths 2–9).

```powershell
.\build\benchmark.exe --input ..\pairs.json --output results\phase2a.csv --iterations 100000 --repeats 5
```

Flags:

- `--input` — path to `pairs.json`
- `--output` — CSV path (default `results/phase2a.csv`)
- `--iterations` — timed loop count per pair/method (default 100000)
- `--repeats` — timing repeats (default 5)
- `--run-id` — optional run id prefix

## End-to-end pipeline

From the **repository root** (Windows; uses `mingw32-make` via CMake):

```powershell
.\phase2a\run_pipeline.ps1
```

Or step by step:

```powershell
# 1. Export pairs
python phase2a/tools/export_pairs.py

# 2. Build and test (from phase2a/build)
cd phase2a
mkdir build -Force; cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
cmake --build . --target run_tests
# Expect: All digit tests passed, algorithm PASS lines, Summary: 20/20 passed

# 3. Benchmark (only if step 2 is green)
cmake --build . --target benchmark
.\benchmark.exe --input ..\..\pairs.json --output ..\results\phase2a.csv

# 4. Compare (re-run Phase 1 on pairs.json first for a fair match)
cd ..\..
python vedic_benchmark/tools/run_from_pairs.py --pairs pairs.json --iterations 100000 --repeat 5
python phase2a/tools/compare_phases.py `
  --phase1 vedic_benchmark/results/benchmark_results.csv `
  --phase2 phase2a/results/phase2a.csv
```

`compare_phases.py` also accepts `--py-csv` / `--cpp-csv` (same as `--phase1` / `--phase2`).

See [VERIFICATION.md](VERIFICATION.md) for counter parity, `vs_ratio`, and `pairs.json` checksum.

## Layout

```
phase2a/
  src/           headers: digits, vedic, schoolbook, native, depth
  tests/         test_digits, test_algorithms, verify_counters
  tools/         compare_phases.py, export_pairs.py, build_counter_parity.py
  verification/  counter_parity.json (golden counts from Python)
  results/       CSV output (gitignored except .gitkeep)
```
