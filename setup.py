"""
Setup script for Sentiment Analysis Application
Run this script to set up the environment and download required dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_backend():
    """Set up Python backend"""
    print("\n" + "="*60)
    print("🔧 Setting up Backend (Python/Flask)")
    print("="*60)
    
    # Install Python packages
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  Warning: spaCy model download failed. You can download it manually later.")
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Warning: NLTK data download failed: {e}")
    
    return True

def setup_frontend():
    """Set up React frontend"""
    print("\n" + "="*60)
    print("🔧 Setting up Frontend (React)")
    print("="*60)
    
    # Check if node is installed
    try:
        result = subprocess.run("node --version", shell=True, capture_output=True, text=True)
        print(f"✓ Node.js {result.stdout.strip()} detected")
    except:
        print("❌ Node.js is not installed. Please install Node.js 14+ from https://nodejs.org/")
        return False
    
    # Navigate to frontend directory
    frontend_dir = os.path.join(os.getcwd(), 'frontend')
    if not os.path.exists(frontend_dir):
        print("❌ Frontend directory not found")
        return False
    
    # Install npm packages
    os.chdir(frontend_dir)
    if not run_command("npm install", "Installing React dependencies"):
        os.chdir('..')
        return False
    
    os.chdir('..')
    return True

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🚀 Sentiment Analysis Application - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup backend
    if not setup_backend():
        print("\n❌ Backend setup failed")
        sys.exit(1)
    
    # Setup frontend
    frontend_success = setup_frontend()
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Start the backend server:")
    print("   python app.py")
    print("\n2. Start the frontend (in a new terminal):")
    print("   cd frontend")
    print("   npm start")
    print("\n3. Open http://localhost:3000 in your browser")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()


