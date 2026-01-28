#!/bin/bash

# Vision AI Agent Startup Script

echo "======================================================================"
echo "🚀 Vision AI Agent - Starting Server"
echo "======================================================================"
echo ""
echo "📸 Vision Model: Florence-2"
echo "🔒 Local Secure Server"
echo "🧠 AI Agent - Self-Modifiable"
echo ""
echo "======================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Check if virtual environment should be used
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Starting Flask server..."
echo "======================================================================"
echo ""
echo "📍 Server will be available at: http://localhost:5000"
echo "📝 Note: First run will download Florence-2 model (~900MB)"
echo ""

# Start the server
python app.py
