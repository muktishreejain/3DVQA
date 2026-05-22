# Windows PowerShell demo launcher
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python scripts\run_demo.py
