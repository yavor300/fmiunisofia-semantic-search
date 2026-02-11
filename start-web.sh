#!/bin/bash

# Startup script for Semantic Search Web Interface

echo "============================================"
echo "Semantic Search Engine - Web Interface"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Check if spaCy model is installed
echo "Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "spaCy model not found. Installing..."
    python -m spacy download en_core_web_sm
    echo "spaCy model installed"
fi

echo ""
echo "============================================"
echo "Starting Flask Web Application"
echo "============================================"
echo ""
echo "The application will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py
