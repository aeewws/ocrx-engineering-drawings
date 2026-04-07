param(
    [string]$ReleaseZipUrl = "https://github.com/aeewws/ocrx-engineering-drawings/releases/latest/download/ocrx-engineering-drawings-windows.zip"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workRoot = Join-Path $env:TEMP ("ocrx-engineering-drawings-" + [guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $workRoot "ocrx-engineering-drawings-windows.zip"
$extractPath = Join-Path $workRoot "extract"

New-Item -ItemType Directory -Force -Path $workRoot, $extractPath | Out-Null

Write-Host "Downloading latest OCRX Engineering Drawings release..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $ReleaseZipUrl -OutFile $zipPath

Write-Host "Extracting package..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

$installer = Get-ChildItem -Path $extractPath -Recurse -Filter "install-codex.ps1" | Select-Object -First 1
if (-not $installer) {
    throw "Could not find scripts\\install-codex.ps1 inside the downloaded package."
}

Write-Host "Running installer..." -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File $installer.FullName
