@echo off
setlocal
cd /d "%~dp0"

echo Starting Rikuy Webcam Viewer...
python rikuy.py
if errorlevel 1 (
    echo.
    echo python.exe failed, trying py launcher...
    py -3 rikuy.py
)

echo.
echo Rikuy Webcam Viewer closed.
pause
