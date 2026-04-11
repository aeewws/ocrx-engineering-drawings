param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
$tempRoot = Join-Path $env:TEMP ("ocrx-install-online-test-" + [Guid]::NewGuid().ToString('N'))
$serverRoot = Join-Path $tempRoot 'server'
$packageRoot = Join-Path $serverRoot 'ocrx-engineering-drawings'
$scriptsDir = Join-Path $packageRoot 'scripts'
$zipPath = Join-Path $serverRoot 'package.zip'
$logPath = Join-Path $tempRoot 'installer-args.json'
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$serverProcess = $null
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonExe = $pythonCommand.Source
}

if (-not $pythonExe) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $pythonExe = $pyLauncher.Source
    }
}

if (-not $pythonExe) {
    throw 'Python was not found on PATH for the local HTTP server smoke test.'
}

try {
    New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null

    @"
param(
    [string]`$CodexHome,
    [string]`$CodexBin,
    [string]`$EnvName,
    [switch]`$SkipAgentsHint
)

`$payload = [pscustomobject]@{
    CodexHome = `$CodexHome
    CodexBin = `$CodexBin
    EnvName = `$EnvName
    SkipAgentsHint = [bool]`$SkipAgentsHint
}
[System.IO.File]::WriteAllText(`$env:TEST_INSTALL_ONLINE_LOG, (`$payload | ConvertTo-Json -Compress), [System.Text.UTF8Encoding]::new(`$false))
"@ | Set-Content -Path (Join-Path $scriptsDir 'install-codex.ps1') -Encoding utf8

    Compress-Archive -Path (Join-Path $packageRoot '*') -DestinationPath $zipPath -CompressionLevel Optimal

    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $listener.Stop()

    $serverArgs = if ([System.IO.Path]::GetFileNameWithoutExtension($pythonExe) -ieq 'py') {
        @('-3.10', '-m', 'http.server', $port, '--bind', '127.0.0.1')
    } else {
        @('-m', 'http.server', $port, '--bind', '127.0.0.1')
    }
    $serverProcess = Start-Process -FilePath $pythonExe -ArgumentList $serverArgs -WorkingDirectory $serverRoot -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 1

    $env:TEST_INSTALL_ONLINE_LOG = $logPath
    & powershell -ExecutionPolicy Bypass -File (Join-Path $repoRoot 'install-online.ps1') `
        -ReleaseZipUrl "http://127.0.0.1:$port/package.zip" `
        -CodexHome 'C:\codex-home' `
        -CodexBin 'C:\codex-bin' `
        -EnvName 'test-env' `
        -SkipAgentsHint

    if (-not (Test-Path -LiteralPath $logPath)) {
        throw 'The packaged installer did not write the expected argument log.'
    }

    $payload = Get-Content -LiteralPath $logPath -Raw | ConvertFrom-Json
    if ($payload.CodexHome -ne 'C:\codex-home') {
        throw 'CodexHome was not forwarded correctly.'
    }
    if ($payload.CodexBin -ne 'C:\codex-bin') {
        throw 'CodexBin was not forwarded correctly.'
    }
    if ($payload.EnvName -ne 'test-env') {
        throw 'EnvName was not forwarded correctly.'
    }
    if (-not $payload.SkipAgentsHint) {
        throw 'SkipAgentsHint was not forwarded correctly.'
    }

    Write-Host 'install-online forwarding smoke passed.'
}
finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
        $serverProcess.WaitForExit()
    }
    if (Test-Path -LiteralPath $tempRoot) {
        Start-Sleep -Milliseconds 250
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
