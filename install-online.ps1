param(
    [string]$ReleaseZipUrl = "https://github.com/aeewws/ocrx-engineering-drawings/releases/latest/download/ocrx-engineering-drawings-windows.zip",
    [string]$CodexHome,
    [string]$CodexBin,
    [string]$EnvName,
    [switch]$SkipAgentsHint
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
$invokeArgs = @("-ExecutionPolicy", "Bypass", "-File", $installer.FullName)
if ($PSBoundParameters.ContainsKey("CodexHome")) {
    $invokeArgs += @("-CodexHome", $CodexHome)
}
if ($PSBoundParameters.ContainsKey("CodexBin")) {
    $invokeArgs += @("-CodexBin", $CodexBin)
}
if ($PSBoundParameters.ContainsKey("EnvName")) {
    $invokeArgs += @("-EnvName", $EnvName)
}
if ($SkipAgentsHint) {
    $invokeArgs += "-SkipAgentsHint"
}

& powershell @invokeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Installer exited with code $LASTEXITCODE."
}
