$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

Set-Location -LiteralPath $projectRoot
& $pythonPath PyWALL_v13.py
exit $LASTEXITCODE
