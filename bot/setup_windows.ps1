# Creates .\venv for the bot and installs its runtime dependencies.
# Run from this folder in PowerShell:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path policy.pt)) {
    Write-Warning "policy.pt is missing. Export one from the training machine first (bin/export)."
}
Write-Host "Done. Start a match with: .\venv\Scripts\python.exe run_match.py match_1v1.toml"
