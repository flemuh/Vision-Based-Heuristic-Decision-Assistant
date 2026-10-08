@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

if not exist "%PROJECT_ROOT%\run_windows.bat" (
  echo [ERROR] Could not resolve Speedlora project root:
  echo   %PROJECT_ROOT%
  pause
  exit /b 2
)

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "SRC=%LOCALAPPDATA%\Speedlora\JewelBingo-V1826-StateIntegrity"
set "DST=%PROJECT_ROOT%\userdata"
set "TMP=%PROJECT_ROOT%\userdata.__migrating__"
set "VERIFY_LOG=%TEMP%\speedlora_portable_verify_%RANDOM%.txt"

cls
echo ============================================================
echo Speedlora - one-time portable profile migration
echo ============================================================
echo.
echo Source:
echo   %SRC%
echo.
echo Destination:
echo   %DST%
echo.
echo This is COPY-ONLY. LocalAppData is never deleted.
echo Close the Assistant before continuing.
echo.

if not exist "%SRC%\" (
  echo [ERROR] Current profile was not found.
  echo Nothing was changed.
  pause
  exit /b 3
)

if exist "%DST%\" (
  dir /b "%DST%" 2^>nul | findstr . ^>nul
  if not errorlevel 1 (
    echo [ERROR] userdata already exists and is not empty.
    echo Refusing to merge two profiles.
    pause
    exit /b 4
  )
)

choice /C YN /N /M "Assistant is CLOSED and you want to COPY the current profile? [Y/N]: "
if errorlevel 2 exit /b 0

if exist "%TMP%\" rmdir /s /q "%TMP%"
mkdir "%TMP%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Could not create temporary destination.
  pause
  exit /b 5
)

echo [1/4] Copying complete profile...
robocopy "%SRC%" "%TMP%" /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XJ /NP /NFL /NDL
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 goto :COPY_FAILED

echo [2/4] Verifying source and copied tree...
robocopy "%SRC%" "%TMP%" /MIR /L /R:0 /W:0 /XJ /NP /NFL /NDL /NJH /NJS > "%VERIFY_LOG%"
set "VRC=%ERRORLEVEL%"
if not "%VRC%"=="0" goto :VERIFY_FAILED

if exist "%DST%\" rmdir /s /q "%DST%"
move "%TMP%" "%DST%" >nul
if errorlevel 1 (
  echo [ERROR] Verification passed, but final rename failed.
  echo Verified temporary copy remains at:
  echo   %TMP%
  pause
  exit /b 6
)

echo [3/4] Checking critical profile files...
if exist "%DST%\bingo.sqlite3" (
  echo       bingo.sqlite3: OK
) else (
  echo [WARNING] bingo.sqlite3 was not found at the profile root.
)
if exist "%DST%\.persistent_data_v1" (
  echo       persistent profile marker: OK
) else (
  echo [WARNING] .persistent_data_v1 marker not found.
)

> "%DST%\PORTABLE_PROFILE_INFO.txt" echo Speedlora portable profile copied from:
>>"%DST%\PORTABLE_PROFILE_INFO.txt" echo %SRC%
>>"%DST%\PORTABLE_PROFILE_INFO.txt" echo.
>>"%DST%\PORTABLE_PROFILE_INFO.txt" echo Original LocalAppData profile was NOT deleted.
>>"%DST%\PORTABLE_PROFILE_INFO.txt" echo Use RUN_PORTABLE.bat from the project root.

del "%VERIFY_LOG%" >nul 2>&1

echo [4/4] Portable profile is ready.
echo.
echo ============================================================
echo MIGRATION COMPLETE - NO ORIGINAL DATA WAS DELETED
echo ============================================================
echo Start from now on with:
echo   %PROJECT_ROOT%\RUN_PORTABLE.bat
echo.
echo After validating a few games, this migration folder is disposable.
pause
exit /b 0

:COPY_FAILED
echo [ERROR] Robocopy failed with code %RC%.
echo Nothing was removed from LocalAppData.
if exist "%TMP%\" rmdir /s /q "%TMP%"
pause
exit /b %RC%

:VERIFY_FAILED
echo [ERROR] Copy verification found differences ^(robocopy code %VRC%^).
echo LocalAppData was not modified. Temporary copy remains:
echo   %TMP%
type "%VERIFY_LOG%"
pause
exit /b 7
