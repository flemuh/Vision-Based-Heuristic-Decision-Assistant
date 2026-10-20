@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.12"
    if not defined PY_CMD py -3.11 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.11"
    if not defined PY_CMD py -3 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1 && set "PY_CMD=py -3"
)
if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1 && set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    echo ERROR: Python 3.11+ not found.
    pause
    exit /b 1
)

echo Installing benchmark dependencies...
%PY_CMD% -m pip install -r requirements-benchmark.txt
if errorlevel 1 (
    echo ERROR: failed to install benchmark dependencies.
    pause
    exit /b 1
)

echo Running heuristic simulation benchmark...
%PY_CMD% tools\benchmark_compare.py --output benchmark-results\compare
if errorlevel 1 (
    echo ERROR: benchmark failed.
    pause
    exit /b 1
)

echo.
echo Benchmark finished.
echo PNG:  benchmark-results\compare\benchmark_compare.png
echo CSV:  benchmark-results\compare\benchmark_compare.csv
echo JSON: benchmark-results\compare\benchmark_compare.json
echo.
pause
endlocal
