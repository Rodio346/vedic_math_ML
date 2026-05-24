# Full reproduction pipeline (run from repository root)
#   .\scripts\run_final_pipeline.ps1
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot | Split-Path -Parent
Set-Location $Root

$LogPath = Join-Path $PSScriptRoot ("final_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $LogPath -Append | Out-Null
Write-Host "Logging to $LogPath" -ForegroundColor DarkGray

$CpuWorkers = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$Phase1Workers = [Math]::Max(4, $CpuWorkers - 1)
Write-Host "Phase 1 pair workers: $Phase1Workers (logical CPUs: $CpuWorkers)" -ForegroundColor DarkGray

$MingwBin = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if (Test-Path $MingwBin) {
    $env:Path = "$MingwBin;C:\Program Files\CMake\bin;" + $env:Path
}

function Test-CsvRowCount {
    param([string]$Path, [int]$ExpectedDataRows)
    if (-not (Test-Path $Path)) { return $false }
    $lines = (Get-Content $Path | Measure-Object -Line).Lines
    return ($lines -ge ($ExpectedDataRows + 1))
}

$Artifacts = @(
    "vedic_benchmark/results/benchmark_results.csv",
    "vedic_benchmark/results/benchmark_detail.jsonl",
    "phase2a/results/phase2a.csv",
    "phase2a/results/comparison.csv",
    "phase2b/results/phase2b.csv",
    "phase2b/results/phase2b_summary.csv",
    "phase3/results/phase3.csv",
    "phase3/results/prediction_vs_actual.csv"
)

function Clear-ResultCsvs {
    Write-Host "=== Clean previous result CSVs ===" -ForegroundColor Cyan
    foreach ($rel in $Artifacts) {
        $p = Join-Path $Root $rel
        if (-not (Test-Path $p)) { continue }
        try {
            Remove-Item -Force $p -ErrorAction Stop
            Write-Host "  removed $rel"
        } catch {
            Write-Host "  skip (locked): $rel" -ForegroundColor Yellow
        }
    }
}

# --- Step 0: pairs ---
Write-Host "=== 0. Export pairs ===" -ForegroundColor Cyan
python phase2a/tools/export_pairs.py

# --- Step 1: Phase 1 tests ---
Write-Host "=== 1. Phase 1 tests ===" -ForegroundColor Cyan
python -m pytest tests/ -v --tb=short
if ($LASTEXITCODE -ne 0) { throw "Phase 1 tests failed" }

# --- Step 2: Phase 1 benchmark ---
$P1Csv = Join-Path $Root "vedic_benchmark\results\benchmark_results.csv"
if (Test-CsvRowCount -Path $P1Csv -ExpectedDataRows 480) {
    Write-Host "=== 2. Phase 1 benchmark - SKIP complete CSV ===" -ForegroundColor Yellow
} else {
    Clear-ResultCsvs
    Write-Host "=== 2. Phase 1 benchmark | pairs.json | 100k x5 | workers=$Phase1Workers ===" -ForegroundColor Cyan
    python -u vedic_benchmark/tools/run_from_pairs.py `
        --pairs-file pairs.json `
        --iterations 100000 `
        --repeat 5 `
        --workers $Phase1Workers
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 benchmark failed" }
    if (-not (Test-CsvRowCount -Path $P1Csv -ExpectedDataRows 480)) {
        throw "Phase 1 CSV incomplete (expected 481 lines including header)"
    }
    $p1Lines = (Get-Content $P1Csv | Measure-Object -Line).Lines
    Write-Host "Phase 1 done: $p1Lines lines in benchmark_results.csv" -ForegroundColor Green
}

# --- Step 3: Phase 2A ---
$P2aCsv = Join-Path $Root "phase2a\results\phase2a.csv"
if (Test-CsvRowCount -Path $P2aCsv -ExpectedDataRows 480) {
    Write-Host "=== 3-4. Phase 2A - SKIP complete CSV ===" -ForegroundColor Yellow
} else {
    Write-Host "=== 3. Phase 2A build + tests + benchmark ===" -ForegroundColor Cyan
    $P2aBuild = Join-Path $Root "phase2a\build"
    if (-not (Test-Path $P2aBuild)) { New-Item -ItemType Directory -Path $P2aBuild | Out-Null }
    Set-Location (Join-Path $Root "phase2a\build")
    cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
    cmake --build . --target run_tests -j $Phase1Workers
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A tests failed" }
    cmake --build . --target benchmark -j $Phase1Workers
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A benchmark build failed" }
    .\benchmark.exe --input ..\..\pairs.json --output ..\results\phase2a.csv --iterations 100000 --repeats 5
    if ($LASTEXITCODE -ne 0) { throw "Phase 2A benchmark run failed" }

    Set-Location $Root
    Write-Host "=== 4. Phase 2A compare ===" -ForegroundColor Cyan
    python phase2a/tools/compare_phases.py `
        --phase1 vedic_benchmark/results/benchmark_results.csv `
        --phase2 phase2a/results/phase2a.csv
}

Set-Location $Root
$P2Compare = Join-Path $Root "phase2a\results\comparison.csv"
if (-not (Test-Path $P2Compare) -and (Test-Path $P2aCsv)) {
    Write-Host "=== 4. Phase 2A compare (resume) ===" -ForegroundColor Cyan
    python phase2a/tools/compare_phases.py `
        --phase1 vedic_benchmark/results/benchmark_results.csv `
        --phase2 phase2a/results/phase2a.csv
}

# --- Step 5: Phase 2B ---
$P2bCsv = Join-Path $Root "phase2b\results\phase2b.csv"
$P2bSummary = Join-Path $Root "phase2b\results\phase2b_summary.csv"
if ((Test-Path $P2bCsv) -and (Test-Path $P2bSummary)) {
    Write-Host "=== 5. Phase 2B - SKIP outputs exist ===" -ForegroundColor Yellow
} else {
    Write-Host "=== 5. Phase 2B tests + DAG run ===" -ForegroundColor Cyan
    Set-Location (Join-Path $Root "phase2b")
    python -m pytest tests/ -v --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Phase 2B tests failed" }
    python -u main.py --pairs-json ..\pairs.json --digit-widths 2 3 4 5 6 7 8 9 --output results\phase2b.csv
    if ($LASTEXITCODE -ne 0) { throw "Phase 2B main failed" }
    Set-Location $Root
}

# --- Step 6–7: Phase 3 ---
Set-Location (Join-Path $Root "phase3")
$P3Csv = Join-Path $Root "phase3\results\phase3.csv"
$P3Pred = Join-Path $Root "phase3\results\prediction_vs_actual.csv"
$P3Exe = Join-Path $Root "phase3\build\Release\benchmark.exe"

if ((Test-Path $P3Pred) -and (Test-Path $P3Csv)) {
    Write-Host "=== 6-7. Phase 3 - SKIP complete outputs ===" -ForegroundColor Yellow
} else {
    Write-Host "=== 6. Phase 3 build + validator + benchmark ===" -ForegroundColor Cyan
    if (-not (Test-Path $P3Exe)) {
        cmake -B build -G "Visual Studio 18 2026" -A x64
        if ($LASTEXITCODE -ne 0) { throw "Phase 3 cmake configure failed" }
    } else {
        Write-Host "  Reusing existing phase3/build (incremental)" -ForegroundColor DarkGray
    }
    cmake --build build --config Release -j $Phase1Workers
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 build failed" }
    .\build\Release\validator.exe --input ..\pairs.json
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 validator failed" }
    # 1000 timed iters (README practical run); full paper-style used 10000 + warmup 100
    .\build\Release\benchmark.exe --input ..\pairs.json --digit-widths 4 5 6 7 8 9 --iterations 1000 --warmup 20 --output results\phase3.csv
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 benchmark failed" }

    Write-Host "=== 7. Phase 3 compare_dag ===" -ForegroundColor Cyan
    python analysis\compare_dag.py
    if ($LASTEXITCODE -ne 0) { throw "compare_dag failed" }
}

Set-Location $Root
Write-Host "=== Done. Artifacts ===" -ForegroundColor Green
foreach ($rel in $Artifacts) {
    $p = Join-Path $Root $rel
    if (Test-Path $p) { Write-Host "  OK  $rel" } else { Write-Host "  MISSING  $rel" -ForegroundColor Red }
}

Stop-Transcript | Out-Null
Write-Host "Log saved: $LogPath" -ForegroundColor Green
