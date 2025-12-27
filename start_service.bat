@echo off
echo Starting GuardianAI ML Service...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create models directory
if not exist "models" (
    echo Creating models directory...
    mkdir models
)

REM Train initial model if not exists
if not exist "models\guardian_model.pkl" (
    echo Training initial model...
    python train_model.py
)

REM Start the service
echo.
echo Starting GuardianAI ML Service on port 8001...
echo Press Ctrl+C to stop the service
echo.
python main.py

pause