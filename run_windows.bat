@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Speedlora Jewel Bingo Assistant V18.0.26 State Integrity Root Fix

if not exist "data" mkdir "data"
set "PIP_LOG=%CD%\data\pip-install.log"
set "PY_CMD="
set "PY_TAG="

> "%PIP_LOG%" echo Speedlora V18.0.26 launcher bootstrap log
>> "%PIP_LOG%" echo ==========================================

echo ============================================================
echo  Speedlora Jewel Bingo Assistant - Windows launcher
echo ============================================================
echo.

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3.12"
        set "PY_TAG=312"
    )
    if not defined PY_CMD (
        py -3.11 -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=py -3.11"
            set "PY_TAG=311"
        )
    )
    if not defined PY_CMD (
        py -3.13 -c "import sys" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=py -3.13"
            set "PY_TAG=313"
        )
    )
    if not defined PY_CMD (
        py -3 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=py -3"
            set "PY_TAG=3"
        )
    )
)

if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=python"
            set "PY_TAG=system"
        )
    )
)

if not defined PY_CMD goto :NO_PYTHON

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
rem Root-fix uses an isolated mutable profile by default so older V11/V21/V26
rem adaptive templates/calibration cannot contaminate this A/B validation.
rem Override SPEEDLORA_JEWEL_DATA_DIR before launch if you intentionally want another profile.
if not defined SPEEDLORA_JEWEL_DATA_DIR set "SPEEDLORA_JEWEL_DATA_DIR=%LOCALAPPDATA%\Speedlora\JewelBingo-V1826-StateIntegrity"
if not defined SPEEDLORA_JEWEL_CLEAN_PROFILE set "SPEEDLORA_JEWEL_CLEAN_PROFILE=1"
set "VENV_DIR=%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py%PY_TAG%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_ok_v18_0_26_p5"

echo [1/5] Python detected: %PY_CMD%
%PY_CMD% --version
if errorlevel 1 goto :PYTHON_FAILED

echo       Runtime venv: %VENV_DIR%
if exist ".venv" echo       NOTE: legacy project-local .venv is ignored by this release.

%PY_CMD% -c "import tkinter" >nul 2>&1
if errorlevel 1 goto :NO_TKINTER

echo.
echo [2/5] Validating host pip from %PY_CMD% ...
call :VALIDATE_HOST_PIP
if errorlevel 1 goto :HOST_PIP_FAILED
echo       Host pip: OK

if not exist "%VENV_PY%" (
    echo [3/5] Creating short-path pip-less virtual environment ...
    call :RECREATE_VENV
    if errorlevel 1 goto :VENV_FAILED
) else (
    echo [3/5] Existing short-path virtual environment found.
    call :VALIDATE_VENV_PYTHON
    if errorlevel 1 (
        echo       Existing runtime venv is incomplete. Recreating it ...
        call :RECREATE_VENV
        if errorlevel 1 goto :VENV_FAILED
    )
)

call :VALIDATE_VENV_PYTHON
if errorlevel 1 goto :VENV_FAILED

echo       venv runtime: OK ^(pip inside venv is not required^)

if not exist "%REQ_MARKER%" (
    echo.
    echo [4/5] Installing project dependencies into short-path venv ...
    echo       Installer: host pip -^> %VENV_PY%
    echo       Detailed pip log: %PIP_LOG%
    call :INSTALL_REQUIREMENTS
    if errorlevel 1 goto :REQUIREMENTS_FAILED
    > "%REQ_MARKER%" echo installed
) else (
    echo [4/5] Dependencies were already installed successfully.
)

echo.
echo [5/5] Checking dependencies ...
call :CHECK_CORE_IMPORTS
if errorlevel 1 (
    echo       Dependency marker is stale or packages are incomplete. Reinstalling ...
    if exist "%REQ_MARKER%" del /q "%REQ_MARKER%" >nul 2>&1
    call :FORCE_REINSTALL_REQUIREMENTS
    if errorlevel 1 goto :REQUIREMENTS_FAILED
    > "%REQ_MARKER%" echo installed
    call :CHECK_CORE_IMPORTS
    if errorlevel 1 goto :IMPORT_FAILED
)

echo       Core dependencies and project imports: OK

echo.
echo Checking Ubuntu/WSL persistent solver backend ...
if not defined JEWEL_BINGO_WSL_DISTRO set "JEWEL_BINGO_WSL_DISTRO=Ubuntu"
set "WSL_DISTRO=%JEWEL_BINGO_WSL_DISTRO%"
where wsl.exe >nul 2>&1
if errorlevel 1 (
    echo       WSL not found. The app will use the local Windows compatibility solver.
) else (
    wsl.exe -d "%WSL_DISTRO%" -- python3 --version >nul 2>&1
    if errorlevel 1 (
        echo       WSL distro "%WSL_DISTRO%" does not have python3. Local fallback will be used.
    ) else (
        wsl.exe -d "%WSL_DISTRO%" --cd "%CD%" -- python3 -c "import vision_engine; from vision_engine.solver.service import SolverEngine" >nul 2>&1
        if errorlevel 1 (
            echo       Ubuntu/Python works, but this project could not be imported from its own folder.
            echo       Local Windows fallback will be used.
        ) else (
            echo       WSL distro "%WSL_DISTRO%" + python3 + project import detected.
            echo       V18.0.26 will use this exact folder via WSL --cd; no wslpath is used.
        )
    )
)

echo.
echo Starting Jewel Bingo Assistant V18.0.26 State Integrity Root Fix ...
echo.
"%VENV_PY%" app.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" goto :APP_FAILED

echo.
echo Application closed normally.
goto :SUCCESS

:VALIDATE_HOST_PIP
%PY_CMD% -c "import pip; import pip._internal.commands.install; import pip._vendor.urllib3" >nul 2>&1
if errorlevel 1 exit /b 1
%PY_CMD% -m pip install --help >nul 2>&1
if errorlevel 1 exit /b 1
%PY_CMD% -m pip --help 2>nul | findstr /C:"--python" >nul
if errorlevel 1 exit /b 1
exit /b 0

:VALIDATE_VENV_PYTHON
if not exist "%VENV_PY%" exit /b 1
"%VENV_PY%" -c "import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info >= (3,11)" >nul 2>&1
exit /b %ERRORLEVEL%

:RECREATE_VENV
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
if exist "%VENV_DIR%" (
    echo       ERROR: Could not remove the old runtime venv:
    echo       %VENV_DIR%
    echo       Close Python processes using it and run the BAT again.
    exit /b 1
)
%PY_CMD% -m venv --without-pip "%VENV_DIR%" >> "%PIP_LOG%" 2>&1
if errorlevel 1 exit /b 1
if not exist "%VENV_PY%" exit /b 1
call :VALIDATE_VENV_PYTHON
exit /b %ERRORLEVEL%

:INSTALL_REQUIREMENTS
%PY_CMD% -m pip --python "%VENV_PY%" install -r requirements.txt --log "%PIP_LOG%"
exit /b %ERRORLEVEL%

:FORCE_REINSTALL_REQUIREMENTS
%PY_CMD% -m pip --python "%VENV_PY%" install --force-reinstall -r requirements.txt --log "%PIP_LOG%"
exit /b %ERRORLEVEL%

:CHECK_CORE_IMPORTS
"%VENV_PY%" -c "import numpy, cv2, mss, PIL, joblib, tkinter; from vision_engine.gui import run" >nul 2>&1
exit /b %ERRORLEVEL%

:NO_PYTHON
echo.
echo ERROR: Python 3.11 or newer was not found.
echo Recommended: install Python 3.12 64-bit from python.org.
goto :FAIL

:NO_TKINTER
echo.
echo ERROR: Python was found, but Tkinter is unavailable.
echo Install the standard 64-bit Python from python.org with Tcl/Tk support.
goto :FAIL

:PYTHON_FAILED
echo.
echo ERROR: Python was detected but could not be started.
goto :FAIL

:HOST_PIP_FAILED
echo.
echo ERROR: pip in the base Python installation is unavailable or too old for --python.
echo Run: %PY_CMD% -m pip --version
goto :FAIL

:VENV_FAILED
echo.
echo ERROR: Could not create a healthy short-path runtime venv.
echo Runtime path: %VENV_DIR%
echo Detailed log: %PIP_LOG%
goto :FAIL

:REQUIREMENTS_FAILED
echo.
echo ERROR: A dependency failed to install into the short-path runtime venv.
echo Runtime path: %VENV_DIR%
echo Detailed log: %PIP_LOG%
echo.
echo Please send me the LAST 30-50 lines of:
echo   data\pip-install.log
goto :FAIL

:IMPORT_FAILED
echo.
echo ERROR: Dependencies were installed, but one or more required modules still cannot be imported.
echo Runtime path: %VENV_DIR%
echo Detailed pip log: %PIP_LOG%
goto :FAIL

:APP_FAILED
echo.
echo ERROR: The application exited with code %APP_EXIT%.
echo Copy the Python traceback displayed above and send it to me.
goto :FAIL

:FAIL
echo.
echo ------------------------------------------------------------
echo The window will stay open so you can read the error.
echo ------------------------------------------------------------
pause
endlocal
exit /b 1

:SUCCESS
endlocal
exit /b 0
