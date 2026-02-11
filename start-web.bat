@echo off
REM Startup script for Semantic Search Web Interface (Windows)

echo ============================================
echo Semantic Search Engine - Web Interface
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Warning: Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo Checking dependencies...
pip install -q -r requirements.txt

REM Check if spaCy model is installed
echo Checking spaCy model...
python -c "import spacy; spacy.load('en_core_web_sm')" 2>nul
if errorlevel 1 (
    echo Warning: spaCy model not found. Installing...
    python -m spacy download en_core_web_sm
    echo spaCy model installed
)

echo.
echo ============================================
echo Starting Flask Web Application
echo ============================================
echo.
echo The application will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the Flask app
python app.py

pause
