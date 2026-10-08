@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "PY="
for %%V in (312 311 313 3 system) do (
  if not defined PY if exist "%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py%%V\Scripts\python.exe" set "PY=%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py%%V\Scripts\python.exe"
)
if not defined PY (
  where py >nul 2>&1
  if not errorlevel 1 set "PY=py -3.12"
)
if not defined PY goto NOPY

echo ============================================================
echo Speedlora - Win32 input matrix - STATE-INTEGRITY-P1 base
echo ============================================================
echo Harness clicks for real. Put Paint on top and do not touch the mouse.
echo.
echo Orthogonal matrix examples:
echo   A1: --label A1_mu_closed_mss_off --games 10 --mss-load off
echo   A2: --label A2_mu_open_mss_off --games 10 --mss-load off
echo   B1: --label B1_mu_closed_mss_percall --games 10 --mss-load per-call
echo   B2: --label B2_mu_open_mss_percall --games 10 --mss-load per-call
echo   C1: --label C1_mu_closed_mss_persistent --games 10 --mss-load persistent
echo   C2: --label C2_mu_open_mss_persistent --games 10 --mss-load persistent
echo.
%PY% tools\win32_input_matrix.py %*
echo.
echo Output is under %%LOCALAPPDATA%%\Speedlora\win32_matrix\
pause
exit /b %errorlevel%
:NOPY
echo ERROR: Speedlora runtime Python not found. Run run_windows.bat once first.
pause
exit /b 1
