param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.11 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 virtual-environment creation failed."
    }
}

function Invoke-VenvPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $pythonPath $($Arguments -join ' ')"
    }
}

Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-VenvPython -Arguments @("-m", "pip", "install", "-e", ".[dev]")

if ($Cpu) {
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cpu")
} else {
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121")
}

$env:SAM2_BUILD_CUDA = "0"
Invoke-VenvPython -Arguments @("-m", "pip", "install", "--no-build-isolation", "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4")

Invoke-VenvPython -Arguments @("-c", "import torch, sam2; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Meta SAM 2 import: OK')")
Write-Host "PyWALL v13 environment ready: $venvPath"
