@echo off
setlocal
set MEDAI_HOST=0.0.0.0
set MEDAI_PORT=5000
cd /d "%~dp0WEB-APP"
echo Starting MedAI Village Point Dashboard
echo.
echo Same laptop:
echo   http://127.0.0.1:%MEDAI_PORT%
echo.
echo Other devices on the same hotspot or local network can use:
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
    for /f "tokens=* delims= " %%B in ("%%A") do echo   http://%%B:%MEDAI_PORT%
)
echo.
echo Internet is not required for typed symptom diagnosis after Python dependencies are installed.
echo.
python app.py
endlocal
