@echo off
rem Zeus installer for Windows cmd
rem Usage:
rem   curl -L https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.bat | cmd
rem
rem Downloads the latest zeus release, installs it under %LOCALAPPDATA%\zeus,
rem and adds it to the user PATH.

setlocal
set "REPO=PositiveMinds/Zeus-CLI-releases"
set "INSTALL_DIR=%LOCALAPPDATA%\zeus"
set "TARGET=x86_64-pc-windows-msvc"

echo Installing zeus for target: %TARGET%

rem Resolve the download URL for the latest release zip.
for /f "usebackq delims=" %%r in (`powershell -NoProfile -Command "(Invoke-RestMethod -Uri 'https://api.github.com/repos/%REPO%/releases/latest' -Headers @{Accept='application/vnd.github+json'}).assets | Where-Object { $_.name -eq 'zeus-%TARGET%.zip' } | Select-Object -ExpandProperty browser_download_url"`) do set "URL=%%r"

if "%URL%"=="" (
    echo No prebuilt binary found for target %TARGET%.
    echo Install from source instead: cargo install --git https://github.com/%REPO%
    exit /b 1
)

echo Downloading %URL%
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
curl -L -o "%TEMP%\zeus-%TARGET%.zip" "%URL%"
powershell -NoProfile -Command "Expand-Archive -Path '%TEMP%\zeus-%TARGET%.zip' -DestinationPath '%TEMP%\zeus-extract' -Force"
copy /y "%TEMP%\zeus-extract\zeus.exe" "%INSTALL_DIR%\zeus.exe" >nul
rmdir /s /q "%TEMP%\zeus-extract" 2>nul
del "%TEMP%\zeus-%TARGET%.zip" 2>nul

rem Add to user PATH if not present.
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','User')"`) do set "USER_PATH=%%p"
echo "%USER_PATH%" | findstr /i "%INSTALL_DIR%" >nul 2>&1
if errorlevel 1 (
  setx Path "%USER_PATH%;%INSTALL_DIR%" >nul
  echo Added %INSTALL_DIR% to your user PATH. Open a new terminal to use zeus.
) else (
  echo zeus was already on your PATH.
)

echo.
echo Installed zeus. Run:
echo   zeus doctor
echo   zeus chat "hello" --provider mock
endlocal