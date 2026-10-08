@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Speedlora Jewel Bingo Assistant - Diagnostics

echo ============================================================
echo  Jewel Bingo Assistant - Diagnostics
echo ============================================================
echo Working directory: %CD%
echo.

echo --- Windows ---
ver
echo.

echo --- Python launcher ---
where py 2>nul
py --version 2>nul
py -0p 2>nul
echo.

echo --- python.exe ---
where python 2>nul
python --version 2>nul
echo.

echo --- Virtual environment ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version
    ".venv\Scripts\python.exe" -m pip --version
    echo.
    echo --- Import check ---
    ".venv\Scripts\python.exe" -c "mods=['numpy','cv2','mss','PIL','joblib','pytesseract','tkinter']; import importlib; [(print(m,'OK') if importlib.import_module(m) else None) for m in mods]; from vision_engine.gui import run; print('project import OK')"
) else (
    echo .venv has not been created yet.
)

echo.
echo --- Pip log ---
if exist "data\pip-install.log" (
    echo Found: data\pip-install.log
    echo Open this file and send the last lines if installation failed.
) else (
    echo No pip-install.log yet.
)

echo.
pause
endlocal
