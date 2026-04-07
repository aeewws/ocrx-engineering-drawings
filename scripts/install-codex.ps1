param(
    [string]$CodexHome = (Join-Path $HOME ".codex"),
    [string]$CodexBin = (Join-Path $env:LOCALAPPDATA "OpenAI\Codex\bin"),
    [string]$EnvName = "paddleocr-clean310",
    [switch]$SkipAgentsHint
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-Python310 {
    $candidates = @(
        @("py", "-3.10"),
        @("python"),
        @("python3")
    )
    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $args = if ($candidate.Count -gt 1) { $candidate[1..($candidate.Count - 1)] } else { @() }
        try {
            & $exe @args -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return ,@($exe) + $args
            }
        } catch {
        }
    }
    throw "Python 3.10 was not found. Install Python 3.10 and run this installer again."
}

function Invoke-External {
    param(
        [string]$Exe,
        [string[]]$ArgumentList
    )
    Write-Host "==> $Exe $($ArgumentList -join ' ')" -ForegroundColor Cyan
    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Exe $($ArgumentList -join ' ')"
    }
}

function Write-AsciiFile {
    param(
        [string]$Path,
        [string]$Content
    )
    $directory = Split-Path -Parent $Path
    if ($directory) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    $normalized = ($Content -replace "`r?`n", "`r`n")
    [System.IO.File]::WriteAllText($Path, $normalized, [System.Text.Encoding]::ASCII)
}

function Find-OdaConverter {
    $roots = @(
        "C:\\Program Files\\ODA",
        "C:\\Program Files (x86)\\ODA"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $match = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "ODAFileConverter.exe" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($match) {
            return $match
        }
    }
    return $null
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$toolSource = Join-Path $repoRoot "tools\ocrx.py"
$requirementsPath = Join-Path $repoRoot "requirements-windows.txt"
$agentsSnippetPath = Join-Path $repoRoot "templates\AGENTS.snippet.md"

if (-not (Test-Path $toolSource)) {
    throw "Missing tool source: $toolSource"
}
if (-not (Test-Path $requirementsPath)) {
    throw "Missing requirements file: $requirementsPath"
}

$envDir = Join-Path $CodexHome ("envs\" + $EnvName)
$toolDir = Join-Path $CodexHome "tools\ocrx"
$toolTarget = Join-Path $toolDir "ocrx.py"
$envScriptsDir = Join-Path $envDir "Scripts"
$envPython = Join-Path $envScriptsDir "python.exe"

New-Item -ItemType Directory -Force -Path $CodexHome, $CodexBin, $toolDir | Out-Null

if (-not (Test-Path $envPython)) {
    $pythonSpec = Resolve-Python310
    $launcherExe = $pythonSpec[0]
    $launcherArgs = if ($pythonSpec.Count -gt 1) { $pythonSpec[1..($pythonSpec.Count - 1)] } else { @() }
    Invoke-External -Exe $launcherExe -ArgumentList ($launcherArgs + @("-m", "venv", $envDir))
}

Invoke-External -Exe $envPython -ArgumentList @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-External -Exe $envPython -ArgumentList @("-m", "pip", "uninstall", "-y", "onnxruntime")
Invoke-External -Exe $envPython -ArgumentList @("-m", "pip", "install", "--no-warn-conflicts", "paddlepaddle-gpu==3.3.0", "-f", "https://www.paddlepaddle.org.cn/packages/stable/cu129/")
Invoke-External -Exe $envPython -ArgumentList @("-m", "pip", "install", "--force-reinstall", "--no-warn-conflicts", "onnxruntime-gpu==1.23.2")
Invoke-External -Exe $envPython -ArgumentList @("-m", "pip", "install", "--no-warn-conflicts", "-r", $requirementsPath)
# Install RapidOCR components without letting pip pull the CPU-only onnxruntime package back in.
Invoke-External -Exe $envPython -ArgumentList @("-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-warn-conflicts", "rapidocr-onnxruntime==1.4.4", "rapid-latex-ocr==0.0.9")

Copy-Item -LiteralPath $toolSource -Destination $toolTarget -Force

$ocrxWrapper = @"
@echo off
setlocal
set "PYTHON_EXE=$envPython"
set "PATH=$envScriptsDir;%PATH%"
"%PYTHON_EXE%" "$toolTarget" %*
"@

$wrapperSpecs = @{
    "ocrx.cmd" = $ocrxWrapper
    "ocrsmart.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" auto --profile drawing --detail auto --dpi 300 --min-width 2200 %*
"@
    "ocrbest.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" auto --profile drawing --detail full --dpi 320 --min-width 2600 %*
"@
    "ocrsmartbatch.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" batch --mode auto --recursive --profile drawing --dpi 300 --min-width 2200 %*
"@
    "ocrbestbatch.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" batch --mode auto --recursive --profile drawing --detail full --dpi 320 --min-width 2600 %*
"@
    "ocrdraw.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" drawing --profile drawing --detail auto --dpi 300 --min-width 2200 %*
"@
    "ocrdrawfull.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" drawing --profile drawing --detail full --dpi 320 --min-width 2600 %*
"@
    "ocrdrawbatch.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" batch --mode drawing --recursive --profile drawing --dpi 300 --min-width 2200 %*
"@
    "ocrdrawfullbatch.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" batch --mode drawing --recursive --profile drawing --detail full --dpi 320 --min-width 2600 %*
"@
    "ocrcad.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" cad %*
"@
    "ocrcadbatch.cmd" = @"
@echo off
call "$CodexBin\\ocrx.cmd" batch --mode cad --recursive --profile drawing --dpi 320 --min-width 2600 %*
"@
}

foreach ($name in $wrapperSpecs.Keys) {
    Write-AsciiFile -Path (Join-Path $CodexBin $name) -Content $wrapperSpecs[$name]
}

if (-not $SkipAgentsHint -and (Test-Path $agentsSnippetPath)) {
    $agentsPath = Join-Path $CodexHome "AGENTS.md"
    $snippet = (Get-Content -LiteralPath $agentsSnippetPath -Raw).Trim()
    $existing = if (Test-Path $agentsPath) { Get-Content -LiteralPath $agentsPath -Raw } else { "" }
    if ($existing -notmatch [regex]::Escape($snippet)) {
        $updated = if ([string]::IsNullOrWhiteSpace($existing)) {
            $snippet + "`r`n"
        } else {
            $existing.TrimEnd() + "`r`n`r`n" + $snippet + "`r`n"
        }
        [System.IO.File]::WriteAllText($agentsPath, $updated, [System.Text.UTF8Encoding]::new($false))
    }
}

$odaConverter = Find-OdaConverter
if ($odaConverter) {
    Write-Host "ODA File Converter detected: $odaConverter" -ForegroundColor Green
} else {
    Write-Host "ODA File Converter not detected. DXF is ready now; DWG needs ODA File Converter for conversion." -ForegroundColor Yellow
}

Invoke-External -Exe $envPython -ArgumentList @($toolTarget, "doctor")

Write-Host ""
Write-Host "Installed OCRX Engineering Drawings into:" -ForegroundColor Green
Write-Host "  Codex home: $CodexHome"
Write-Host "  Environment: $envDir"
Write-Host "  Wrapper bin: $CodexBin"
Write-Host ""
Write-Host "Primary commands:" -ForegroundColor Green
Write-Host "  ocrsmart"
Write-Host "  ocrbest"
Write-Host "  ocrsmartbatch"
Write-Host "  ocrbestbatch"
Write-Host "  ocrcad"
