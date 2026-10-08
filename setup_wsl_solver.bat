@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Speedlora Jewel Bingo Assistant V18.0.26 - WSL Solver Check

rem Explicit distro: never use the global WSL default (often docker-desktop).
if not defined JEWEL_BINGO_WSL_DISTRO set "JEWEL_BINGO_WSL_DISTRO=Ubuntu"
set "WSL_DISTRO=%JEWEL_BINGO_WSL_DISTRO%"

 echo ============================================================
 echo  V18.0.26 Ubuntu/WSL Solver Check
 echo ============================================================
 echo Solver distro: %WSL_DISTRO%
 echo Project folder: %CD%
 echo.

where wsl.exe >nul 2>&1
if errorlevel 1 (
  echo ERROR: WSL is not installed or wsl.exe is not on PATH.
  goto :FAIL
)

wsl.exe -d "%WSL_DISTRO%" -- python3 --version
if errorlevel 1 (
  echo ERROR: python3 is unavailable in WSL distro "%WSL_DISTRO%".
  echo        Check installed distros with: wsl -l -v
  goto :FAIL
)

rem WSL itself enters THIS extracted project folder. No wslpath, no Downloads
rem lookup and no dependency on the current/default distro.
echo.
echo Checking this project directly from Ubuntu ...
wsl.exe -d "%WSL_DISTRO%" --cd "%CD%" -- python3 -c "import os, vision_engine; print('WSL project directory:', os.getcwd()); print('Jewel Bingo version:', vision_engine.__version__); from vision_engine.solver.service import SolverEngine; print('WSL solver imports: OK')"
if errorlevel 1 (
  echo.
  echo ERROR: Ubuntu could not open/import this extracted project folder.
  echo        The launcher uses its own folder directly via WSL --cd.
  goto :FAIL
)

echo.
echo WSL solver environment is ready in "%WSL_DISTRO%".
echo Docker Desktop/default WSL distro is not changed by this project.
echo No wslpath or manual Linux project path is required.
echo run_windows.bat will auto-start the persistent solver from this folder.
pause
endlocal
exit /b 0

:FAIL
echo.
echo WSL solver check failed. The Windows local fallback can still run the app.
pause
endlocal
exit /b 1
