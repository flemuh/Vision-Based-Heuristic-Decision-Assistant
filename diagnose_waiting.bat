@echo off
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py312"
set "PY=%VENV%\Scripts\python.exe"
if exist "%PY%" goto RUN
where py >nul 2>&1
if errorlevel 1 goto NOPY
set "PY=py -3.12"
:RUN
echo ============================================================
echo Speedlora Jewel Bingo - WAITING live diagnostic
echo ============================================================
echo.
echo Keep the Assistant OPEN in the stuck WAITING state.
echo Capturing runtime, WSL fence, monitor, bridge, Win32 and threads...
echo.
%PY% tools\capture_waiting_diagnostics.py --seconds 12 --interval 0.5
echo.
echo ------------------------------------------------------------
echo Send the generated waiting_*.zip back for analysis.
echo ------------------------------------------------------------
pause
exit /b %errorlevel%
:NOPY
echo ERROR: Python 3.12 was not found.
pause
exit /b 1
