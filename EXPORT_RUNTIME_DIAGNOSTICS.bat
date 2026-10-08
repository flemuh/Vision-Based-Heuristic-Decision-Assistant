@echo off
setlocal
cd /d "%~dp0"
if not defined SPEEDLORA_JEWEL_DATA_DIR set "SPEEDLORA_JEWEL_DATA_DIR=%LOCALAPPDATA%\Speedlora\JewelBingo-V1826-StateIntegrity"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" tools\export_runtime_diagnostics.py --hours 8
if errorlevel 1 (
  echo.
  echo Diagnostic export failed.
) else (
  echo.
  echo Diagnostic ZIP created in Downloads.
)
pause
endlocal
