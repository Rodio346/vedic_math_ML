# Phase 2A end-to-end pipeline (run from repository root)
#   .\phase2a\run_pipeline.ps1
# Requires: Python 3, CMake, MinGW g++ on PATH (see README.md)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$MingwBin = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if (Test-Path $MingwBin) {
    $env:Path = "$MingwBin;C:\Program Files\CMake\bin;" + $env:Path
}

Set-Location $Root

Write-Host "=== 1. Export pairs ===" -ForegroundColor Cyan
python phase2a/tools/export_pairs.py

Write-Host "=== 2. Build and run tests ===" -ForegroundColor Cyan
$BuildDir = Join-Path $PSScriptRoot "build"
if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }
Set-Location $BuildDir
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
cmake --build . --target run_tests -j 4
if ($LASTEXITCODE -ne 0) { throw "run_tests failed" }

Write-Host "=== 3. Benchmark ===" -ForegroundColor Cyan
cmake --build . --target benchmark -j 4
if ($LASTEXITCODE -ne 0) { throw "benchmark build failed" }
.\benchmark.exe --input ..\..\pairs.json --output ..\results\phase2a.csv
if ($LASTEXITCODE -ne 0) { throw "benchmark run failed" }

Set-Location $Root
Write-Host "=== 4. Phase 1 on same pairs (long; same iterations as C++) ===" -ForegroundColor Cyan
python vedic_benchmark/tools/run_from_pairs.py --pairs pairs.json --iterations 100000 --repeat 5 --workers 4

Write-Host "=== 5. Compare ===" -ForegroundColor Cyan
python phase2a/tools/compare_phases.py `
    --phase1 vedic_benchmark/results/benchmark_results.csv `
    --phase2 phase2a/results/phase2a.csv

Write-Host "Done." -ForegroundColor Green
