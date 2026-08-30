@echo off
setlocal
cd /d "%~dp0"

echo Starting Rikuy Webcam Viewer...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if not errorlevel 1 goto run_venv
)

where python >nul 2>&1
if not errorlevel 1 (
    python --version >nul 2>&1
    if not errorlevel 1 goto run_python
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3 --version >nul 2>&1
    if not errorlevel 1 goto run_py
)

echo.
echo ERROR: No usable Python 3 interpreter was found.
echo Create .venv or install Python, then try again.
set "exit_code=1"
goto launch_failed

:run_venv
".venv\Scripts\python.exe" rikuy.py
set "exit_code=%ERRORLEVEL%"
goto launch_finished

:run_python
python rikuy.py
set "exit_code=%ERRORLEVEL%"
goto launch_finished

:run_py
py -3 rikuy.py
set "exit_code=%ERRORLEVEL%"

:launch_finished
if "%exit_code%"=="0" (
    endlocal
    exit /b 0
)

echo.
echo ERROR: Rikuy exited with code %exit_code%.

:launch_failed
pause
endlocal & exit /b %exit_code%
