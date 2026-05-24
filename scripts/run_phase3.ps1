# Phase 3 only — GPU validator + benchmark + compare_dag
#   .\scripts\run_phase3.ps1
#   .\scripts\run_phase3.ps1 -Iterations 500 -Warmup 10
param(
    [int]$Iterations = 1000,
    [int]$Warmup = 20,
    [int[]]$DigitWidths = @(4, 5, 6, 7, 8, 9)
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot | Split-Path -Parent
Set-Location (Join-Path $Root "phase3")

$CpuWorkers = [Math]::Max(4, (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors - 1)
$P3Exe = Join-Path $Root "phase3\build\Release\benchmark.exe"
$widthArgs = ($DigitWidths | ForEach-Object { "$_" }) -join " "

Write-Host "Phase 3 | iterations=$Iterations warmup=$Warmup widths=$widthArgs" -ForegroundColor Cyan

foreach ($rel in @("results\phase3.csv", "results\prediction_vs_actual.csv")) {
    $p = Join-Path (Get-Location) $rel
    if (-not (Test-Path $p)) { continue }
    try {
        Remove-Item -Force $p -ErrorAction Stop
        Write-Host "  removed $rel"
    } catch {
        Write-Host "  skip delete (locked): $rel - benchmark will overwrite on run" -ForegroundColor Yellow
    }
}

if (-not (Test-Path $P3Exe)) {
    cmake -B build -G "Visual Studio 18 2026" -A x64
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 cmake configure failed" }
}
cmake --build build --config Release -j $CpuWorkers
if ($LASTEXITCODE -ne 0) { throw "Phase 3 build failed" }

.\build\Release\validator.exe --input ..\pairs.json
if ($LASTEXITCODE -ne 0) { throw "Phase 3 validator failed" }

$benchArgs = @(
    "--input", "..\pairs.json",
    "--digit-widths") + ($DigitWidths | ForEach-Object { "$_" }) + @(
    "--iterations", "$Iterations",
    "--warmup", "$Warmup",
    "--output", "results\phase3.csv"
)
& .\build\Release\benchmark.exe @benchArgs
if ($LASTEXITCODE -ne 0) { throw "Phase 3 benchmark failed" }

python analysis\compare_dag.py --digit-widths @DigitWidths
if ($LASTEXITCODE -ne 0) { throw "compare_dag failed" }

Set-Location $Root
Write-Host "Done: phase3/results/phase3.csv, phase3/results/prediction_vs_actual.csv" -ForegroundColor Green
