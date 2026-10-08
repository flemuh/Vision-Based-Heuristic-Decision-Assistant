@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Speedlora Jewel Bingo Assistant V18.0.26 - Setup and Test

if not exist "data" mkdir "data"
set "PIP_LOG=%CD%\data\pip-install.log"
set "PY_CMD="
set "PY_TAG="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (set "PY_CMD=py -3.12"& set "PY_TAG=312")
    if not defined PY_CMD (
        py -3.11 -c "import sys" >nul 2>&1
        if not errorlevel 1 (set "PY_CMD=py -3.11"& set "PY_TAG=311")
    )
    if not defined PY_CMD (
        py -3.13 -c "import sys" >nul 2>&1
        if not errorlevel 1 (set "PY_CMD=py -3.13"& set "PY_TAG=313")
    )
    if not defined PY_CMD (
        py -3 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
        if not errorlevel 1 (set "PY_CMD=py -3"& set "PY_TAG=3")
    )
)
if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
        if not errorlevel 1 (set "PY_CMD=python"& set "PY_TAG=system")
    )
)

if not defined PY_CMD (
    echo ERROR: Python 3.11+ not found.
    goto :FAIL
)

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "VENV_DIR=%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py%PY_TAG%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_MARKER=%VENV_DIR%\.requirements_ok_v18_0_26"

echo Using: %PY_CMD%
%PY_CMD% --version
echo Runtime venv: %VENV_DIR%
%PY_CMD% -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Tkinter is unavailable in this Python installation.
    goto :FAIL
)

call :VALIDATE_HOST_PIP
if errorlevel 1 (
    echo ERROR: base Python pip is unavailable or too old for --python.
    goto :FAIL
)

if not exist "%VENV_PY%" (
    call :RECREATE_VENV
    if errorlevel 1 goto :FAIL
) else (
    call :VALIDATE_VENV_PYTHON
    if errorlevel 1 (
        echo Existing short-path runtime venv is incomplete. Recreating it ...
        call :RECREATE_VENV
        if errorlevel 1 goto :FAIL
    )
)

call :INSTALL_REQUIREMENTS
if errorlevel 1 goto :FAIL
> "%REQ_MARKER%" echo installed

"%VENV_PY%" -m pytest -q
if errorlevel 1 goto :FAIL

echo.
echo Setup and tests completed successfully.
pause
endlocal
exit /b 0

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
if exist "%VENV_DIR%" exit /b 1
%PY_CMD% -m venv --without-pip "%VENV_DIR%" >> "%PIP_LOG%" 2>&1
if errorlevel 1 exit /b 1
call :VALIDATE_VENV_PYTHON
exit /b %ERRORLEVEL%

:INSTALL_REQUIREMENTS
%PY_CMD% -m pip --python "%VENV_PY%" install -r requirements.txt --log "%PIP_LOG%"
exit /b %ERRORLEVEL%

:FAIL
echo.
echo Setup failed. The window will remain open.
echo Runtime venv: %VENV_DIR%
echo pip log: %PIP_LOG%
pause
endlocal
exit /b 1
