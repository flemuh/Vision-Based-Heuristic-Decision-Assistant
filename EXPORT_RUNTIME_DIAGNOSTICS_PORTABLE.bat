@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0userdata\" (
  echo [ERROR] Portable profile not found: %~dp0userdata
  pause
  exit /b 2
)
set "SPEEDLORA_JEWEL_DATA_DIR=%~dp0userdata"
if exist "%~dp0EXPORT_RUNTIME_DIAGNOSTICS.bat" (
  call "%~dp0EXPORT_RUNTIME_DIAGNOSTICS.bat"
) else (
  echo [ERROR] EXPORT_RUNTIME_DIAGNOSTICS.bat is not present in this project.
  pause
  exit /b 3
)
endlocal
