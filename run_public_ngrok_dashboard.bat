@echo off
setlocal
set MEDAI_HOST=0.0.0.0
set MEDAI_PORT=5000
set MEDAI_NGROK_DOMAIN=demetra-varietal-cindi.ngrok-free.dev

start "MedAI Dashboard Server" cmd /k "cd /d ""%~dp0WEB-APP"" && set MEDAI_HOST=%MEDAI_HOST%&& set MEDAI_PORT=%MEDAI_PORT%&& python app.py"
timeout /t 8 /nobreak >nul

echo Opening public tunnel:
echo   https://%MEDAI_NGROK_DOMAIN%
echo.
ngrok http --domain=%MEDAI_NGROK_DOMAIN% %MEDAI_PORT%
endlocal
