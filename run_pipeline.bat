@echo off
rem ==========================================================================
rem  Retrain everything: SQL layer, 5 models, threshold, SHAP scores,
rem  figures and the PDF report (~12 minutes on a 4-core laptop).
rem  Optional: set ANTHROPIC_API_KEY first to have Claude write the briefs.
rem ==========================================================================
setlocal
title Churn Intelligence - training pipeline
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Run run_dashboard.bat once first - it sets up the Python environment.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -c "import reportlab" >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing development extras ^(notebooks, tests, PDF report^)...
    "venv\Scripts\python.exe" -m pip install -r requirements-dev.txt || goto :failed
)

"venv\Scripts\python.exe" -m src.pipeline %* || goto :failed
echo.
echo   Done. Start the dashboard with run_dashboard.bat
pause
exit /b 0

:failed
echo.
echo   The pipeline stopped with an error - see the messages above.
pause
exit /b 1
