@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0userdata\" (
  echo [ERROR] Portable profile not found: %~dp0userdata
  echo Run migration\portable_profile\MIGRATE_EXISTING_PROFILE_TO_PORTABLE.bat first.
  pause
  exit /b 2
)
set "SPEEDLORA_JEWEL_DATA_DIR=%~dp0userdata"
call "%~dp0run_windows.bat"
endlocal
