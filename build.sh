#!/bin/bash

# Build script for Render deployment

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

echo "Installing frontend dependencies..."
cd frontend
npm install

echo "Building React app..."
npm run build

echo "Build complete!"

