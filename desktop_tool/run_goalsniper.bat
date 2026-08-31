@echo off
cd /d "%~dp0"

echo Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python was not found on your PATH.
    echo Install it from https://www.python.org/downloads/ and make sure to
    echo check "Add python.exe to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)

echo Installing dependencies (this shows any real error, unlike before)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo pip install failed - see the error above.
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Goalsniper...
python goalsniper_desktop.py
if errorlevel 1 (
    echo.
    echo Goalsniper closed with an error - see above.
)

echo.
echo Press any key to close this window.
pause >nul
