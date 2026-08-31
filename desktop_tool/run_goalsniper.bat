@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on your PATH.
    echo Install it from https://www.python.org/downloads/ and make sure to
    echo check "Add python.exe to PATH" during setup, then run this again.
    pause
    exit /b 1
)

echo Checking dependencies...
pip install -r requirements.txt >nul 2>nul

start "" pythonw goalsniper_desktop.py
