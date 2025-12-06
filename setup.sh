#!/bin/bash

# Setup script for Shark Tank Pitch Analyzer

echo "Setting up Shark Tank Pitch Analyzer..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "Setup complete!"
echo ""
echo "To use the analyzer:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run: python main.py <audio_file_path>"
echo ""
echo "Optional: Create a .env file with your OPENAI_API_KEY for enhanced feedback"
