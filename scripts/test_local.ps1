param(
    [switch]$Sam2
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m compileall -q PyWALL_v13.py sam2_segmentation.py tests
if ($LASTEXITCODE -ne 0) {
    throw "Python compilation checks failed."
}

if ($Sam2) {
    $env:PYWALL_RUN_SAM2_TEST = "1"
} else {
    Remove-Item Env:PYWALL_RUN_SAM2_TEST -ErrorAction SilentlyContinue
}

& $pythonPath -m pytest -p no:cacheprovider -q
exit $LASTEXITCODE
