@echo off
rem Zeus installer for Windows cmd
rem Usage:
rem   curl -L https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.bat | cmd
rem
rem Downloads the latest zeus release, installs it under %LOCALAPPDATA%\zeus,
rem and adds it to the user PATH.

setlocal
set "INSTALL_DIR=%LOCALAPPDATA%\zeus"
set "TARGET=x86_64-pc-windows-msvc"

rem Stable URL that redirects to the latest release's Windows asset.
set "URL=https://github.com/PositiveMinds/Zeus-CLI-releases/releases/latest/download/zeus-%TARGET%.zip"

echo Installing zeus for target: %TARGET%
echo Downloading latest zeus release...
echo   %URL%

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
curl -fL -o "%TEMP%\zeus-%TARGET%.zip" "%URL%"
if errorlevel 1 goto :fail

echo Extracting...
powershell -NoProfile -Command "Expand-Archive -Path '%TEMP%\zeus-%TARGET%.zip' -DestinationPath '%TEMP%\zeus-extract' -Force"
if errorlevel 1 goto :fail

copy /y "%TEMP%\zeus-extract\zeus.exe" "%INSTALL_DIR%\zeus.exe" >nul
if errorlevel 1 goto :fail
rmdir /s /q "%TEMP%\zeus-extract" 2>nul
del "%TEMP%\zeus-%TARGET%.zip" 2>nul

echo.
echo Installed zeus. Verify:
echo   %INSTALL_DIR%\zeus.exe doctor
echo   %INSTALL_DIR%\zeus.exe chat "hello"
echo.
echo Add %INSTALL_DIR% to your user PATH to run just "zeus":
setx Path "%INSTALL_DIR%;%PATH%" >nul
echo Added to PATH. Open a new terminal to use "zeus".
endlocal
exit /b 0

:fail
echo.
echo Error: zeus installation failed. The latest release may not exist yet.
echo Check https://github.com/PositiveMinds/Zeus-CLI-releases/releases
echo or install from source: cargo install --git https://github.com/PositiveMinds/Zeus-CLI
endlocal
exit /b 1