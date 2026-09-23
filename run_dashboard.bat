@echo off
rem ==========================================================================
rem  Churn Intelligence - one-click launcher (Windows)
rem  Double-click this file. First run sets everything up (a few minutes);
rem  after that the dashboard opens in your browser in seconds.
rem ==========================================================================
setlocal
title Churn Intelligence Dashboard
cd /d "%~dp0"

call :ensure_env || goto :failed

echo.
echo   Churn Intelligence is starting at http://localhost:8501
echo   Your browser will open automatically. Close this window to stop the app.
echo.
rem Open the browser once the server has had a few seconds to start
rem (set NO_BROWSER=1 to skip, e.g. on a server).
if not defined NO_BROWSER start "" /min cmd /c "timeout /t 6 /nobreak >nul & start "" http://localhost:8501"
"venv\Scripts\python.exe" -m streamlit run dashboard\app.py --server.headless true
goto :eof

:ensure_env
if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating a Python virtual environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 -m venv venv 2>nul || py -3 -m venv venv
    ) else (
        python -m venv venv
    )
    if not exist "venv\Scripts\python.exe" (
        echo [error] Python 3.11+ was not found. Install it from https://www.python.org/downloads/
        exit /b 1
    )
)
"venv\Scripts\python.exe" -c "import streamlit, xgboost, shap, plotly" >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing requirements - first run only, this takes a few minutes...
    "venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    "venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
)
exit /b 0

:failed
echo.
echo   Setup failed - see the messages above.
pause
exit /b 1
