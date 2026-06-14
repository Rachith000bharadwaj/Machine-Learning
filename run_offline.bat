@echo off
cd /d "%~dp0WEB-APP"
set MEDAI_HOST=0.0.0.0
set MEDAI_PORT=5000
echo Starting MedAI Diagnosis Assistant in offline/local mode...
echo.
echo Open on this computer:
echo   http://127.0.0.1:5000
echo.
echo For another phone/laptop, connect it to the same hotspot or Wi-Fi.
echo The app will also print this computer's local address below.
echo.
python app.py
pause
