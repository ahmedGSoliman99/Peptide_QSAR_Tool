@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys,struct; raise SystemExit(0 if (sys.version_info >= (3,10) and struct.calcsize('P')*8 == 64) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Existing .venv is not Python 3.10+ 64-bit. Rebuilding environment ...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_PY%" (
    echo Creating local environment in %VENV_DIR% ...
    call :create_venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        echo Install Python 3.10+ 64-bit from https://www.python.org/downloads/windows/
        pause
        exit /b 1
    )
)

echo Checking required Python packages ...
"%VENV_PY%" -c "import streamlit, pandas, numpy, sklearn, Bio, plotly, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages. First run may take several minutes ...
    "%VENV_PY%" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 (
        echo Failed to update pip tooling.
        pause
        exit /b 1
    )

    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed.
        echo If this machine has only 32-bit Python, please install 64-bit Python.
        pause
        exit /b 1
    )
)

echo Starting Peptide QSAR Prediction Tool...
start "" "http://localhost:8501"
"%VENV_PY%" -m streamlit run app.py --global.developmentMode false --server.headless true --server.port 8501 --browser.gatherUsageStats false
exit /b 0

:create_venv
for %%V in (3.13 3.12 3.11 3.10 3.14) do (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        py -%%V -m venv "%VENV_DIR%"
        if not errorlevel 1 exit /b 0
    )
)

python -c "import sys" >nul 2>nul
if not errorlevel 1 (
    python -m venv "%VENV_DIR%"
    if not errorlevel 1 exit /b 0
)

exit /b 1
