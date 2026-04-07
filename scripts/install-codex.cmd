@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0install-codex.ps1" %*
