@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
if not defined SPEEDLORA_JEWEL_DATA_DIR set "SPEEDLORA_JEWEL_DATA_DIR=%LOCALAPPDATA%\Speedlora\JewelBingo-V1826-StateIntegrity"
set "PY="
for %%V in (312 311 313 3 system) do (
  if not defined PY if exist "%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py%%V\Scripts\python.exe" set "PY=%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py%%V\Scripts\python.exe"
)
if not defined PY (
  where py >nul 2>&1 && set "PY=py -3.12"
)
if not defined PY goto NOPY
%PY% tools\capture_win32_boundary_diagnostics.py
pause
exit /b %errorlevel%
:NOPY
echo ERROR: Speedlora runtime Python not found. Run run_windows.bat once first.
pause
exit /b 1
