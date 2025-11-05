# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
python setup.py
```

### Option 2: Manual Setup

#### Step 1: Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (optional, will download automatically)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

#### Step 2: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Go back to root
cd ..
```

### Step 3: Run the Application

#### Terminal 1 - Backend:
```bash
python app.py
```
Backend runs on: http://localhost:5000

#### Terminal 2 - Frontend:
```bash
cd frontend
npm start
```
Frontend runs on: http://localhost:3000

### Step 4: Access the Application

Open your browser and go to: **http://localhost:3000**

## 🎯 First Steps

1. **Check Health**: The dashboard will show if the model is ready
2. **Predict Sentiment**: Try entering a review in the "Predict Sentiment" tab
3. **Upload Data**: Upload your CSV file in the "Upload Data" tab
4. **Train Model**: If needed, train the model using existing data

## 📝 Sample CSV Format

Create a CSV file with this structure:

```csv
Review,Rating
"This is a great product!",5
"Terrible service, very disappointed",1
"Average experience, nothing special",3
```

## ⚠️ Troubleshooting

### Backend Issues
- **Port 5000 already in use**: Change port in `app.py` (last line)
- **Model not found**: Upload data and train the model first
- **spaCy error**: Run `python -m spacy download en_core_web_sm`

### Frontend Issues
- **Port 3000 already in use**: React will ask to use a different port
- **npm install fails**: Try `npm install --legacy-peer-deps`
- **CORS errors**: Make sure backend is running on port 5000

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore all features in the dashboard
- Upload your own data and train custom models

