@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Python virtual environment was not found.
    echo Run run_app.bat once first, then run this file again.
    pause
    exit /b 1
)

echo Installing PyInstaller if needed...
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 (
    echo Failed to install PyInstaller.
    pause
    exit /b 1
)

echo Building one-file Windows executable...
"%VENV_PY%" -m PyInstaller --noconfirm --clean --onefile --name PeptideQSAR peptide_qsar_single_file.py --collect-all streamlit --collect-all pandas --collect-all numpy --collect-all sklearn --collect-all scipy --collect-all Bio --collect-all plotly --collect-all matplotlib --collect-all openpyxl --collect-all joblib --hidden-import sklearn.utils._typedefs --hidden-import sklearn.neighbors._partition_nodes
if errorlevel 1 (
    echo EXE build failed.
    pause
    exit /b 1
)

echo.
echo Build finished:
echo %CD%\dist\PeptideQSAR.exe
pause
