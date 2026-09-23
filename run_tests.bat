@echo off
rem Run the automated test suite (economics, SQL layer, scenarios, drift, Copilot, ...).
setlocal
title Churn Intelligence - tests
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Run run_dashboard.bat once first - it sets up the Python environment.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -c "import pytest" >nul 2>nul || "venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
"venv\Scripts\python.exe" -m pytest -q
pause
