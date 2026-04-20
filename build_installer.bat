@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not exist "%ISCC%" (
    echo Inno Setup compiler was not found.
    echo Install it with:
    echo winget install --id JRSoftware.InnoSetup -e
    pause
    exit /b 1
)

if not exist "dist\PeptideQSAR.exe" (
    echo dist\PeptideQSAR.exe was not found.
    echo Run build_exe.bat first.
    pause
    exit /b 1
)

echo Building Windows installer...
"%ISCC%" "installer\PeptideQSAR_Inno.iss"
if errorlevel 1 (
    echo Installer build failed.
    pause
    exit /b 1
)

echo.
echo Installer ready:
echo %CD%\installer\PeptideQSAR_Setup.exe
pause
