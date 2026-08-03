$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$checkpointDir = Join-Path $projectRoot "checkpoints"
$destination = Join-Path $checkpointDir "sam2.1_hiera_base_plus.pt"
$temporary = "$destination.download"
$url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
$expected = "A2345AEDE8715AB1D5D31B4A509FB160C5A4AF1970F199D9054CCFB746C004C5"

New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $destination)) {
    Invoke-WebRequest -Uri $url -OutFile $temporary
    $downloadHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
    if ($downloadHash -ne $expected) {
        Remove-Item -LiteralPath $temporary -Force
        throw "Checkpoint checksum mismatch. Expected $expected; received $downloadHash"
    }
    Move-Item -LiteralPath $temporary -Destination $destination
}

$actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
if ($actual -ne $expected) {
    throw "Existing checkpoint checksum mismatch. Expected $expected; received $actual"
}

Write-Host "Meta SAM 2 checkpoint verified: $destination"
Write-Host "SHA-256: $actual"
