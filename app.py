from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from datetime import datetime

# Import advanced modules
from ml_models import ModelManager, PARAM_GRIDS
from advanced_nlp import AdvancedNLP
from model_interpretability import ModelInterpreter
from database import DatabaseManager
from time_series_analysis import TimeSeriesAnalyzer
from export_utils import ExportManager
from airline_insights import AirlineReviewScraper, AirlineInsightsAnalyzer

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

# Initialize NLTK and spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Global variables for model and vectorizer
model = None
tfidf_vectorizer = None
model_accuracy = 0
model_trained = False

# Initialize advanced modules
model_manager = ModelManager()
nlp_processor = AdvancedNLP()
db_manager = DatabaseManager()
time_analyzer = TimeSeriesAnalyzer()
export_manager = ExportManager()

# Complaint categories
COMPLAINT_KEYWORDS = {
    'Refund Issues': ['refund', 'money back', 'not received', 'reimbursement'],
    'Customer Service': ['support', 'help', 'rude', 'not responding', 'contact'],
    'Booking Issues': ['booking', 'ticket', 'confirmation', 'reservation'],
    'Scam or Fraud': ['scam', 'fraud', 'cheat', 'dishonest', 'fake'],
    'Delays': ['delay', 'late', 'waiting', 'postponed'],
    'Technical Issues': ['website', 'crash', 'app', 'error', 'technical'],
}

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation, numbers, special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    if nlp:
        doc = nlp(" ".join(filtered_tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
        return " ".join(lemmatized_tokens)
    else:
        return " ".join(filtered_tokens)

def assign_category(text):
    """Assign complaint category based on keywords"""
    if isinstance(text, str):
        text = text.lower()
        for category, keywords in COMPLAINT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return category
    return 'Other'

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, tfidf_vectorizer, model_accuracy, model_trained
    
    # Try to load existing model
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            if os.path.exists('model_info.json'):
                with open('model_info.json', 'r') as f:
                    model_info = json.load(f)
                    model_accuracy = model_info.get('accuracy', 0)
            model_trained = True
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Train new model if data exists
    if os.path.exists('reviews_cleaned.csv'):
        try:
            df = pd.read_csv('reviews_cleaned.csv')
            df = df.dropna(subset=['Review', 'Sentiment'])
            
            X = df['Review']
            y = df['Sentiment']
            
            tfidf_vectorizer = TfidfVectorizer()
            X_tfidf = tfidf_vectorizer.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42
            )
            
            # Use class_weight='balanced' to handle imbalanced classes (especially Neutral)
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            model_accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
            
            model_info = {
                'accuracy': float(model_accuracy),
                'trained_date': datetime.now().isoformat()
            }
            with open('model_info.json', 'w') as f:
                json.dump(model_info, f)
            
            model_trained = True
            print(f"Model trained successfully with accuracy: {model_accuracy:.4f}")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    return False

# Initialize model on startup
load_or_train_model()

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model_trained,
        'model_accuracy': model_accuracy
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict sentiment for a single review"""
    if not model_trained:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    data = request.json
    review_text = data.get('review', '')
    
    if not review_text:
        return jsonify({'error': 'Review text is required'}), 400
    
    try:
        # Preprocess and predict
        review_tfidf = tfidf_vectorizer.transform([review_text])
        probabilities = model.predict_proba(review_tfidf)[0]
        classes = model.classes_
        
        # Ensure probabilities is a numpy array (not sparse)
        if hasattr(probabilities, 'toarray'):
            probabilities = probabilities.toarray()[0]
        probabilities = np.array(probabilities).flatten()
        
        # Create probability dict
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        # Improved prediction: Use probability threshold for neutral
        # If max probability is low or neutral has significant probability, check for neutral
        max_prob = float(np.max(probabilities))
        max_idx = int(np.argmax(probabilities))
        predicted_class = classes[max_idx]
        
        # Check if neutral is in classes and has reasonable probability
        if 'Neutral' in classes:
            neutral_idx = list(classes).index('Neutral')
            neutral_prob = probabilities[neutral_idx]
            
            # If neutral probability is high (>0.3) and max prob is not very confident (<0.7)
            # or if neutral is the highest, use neutral
            if (neutral_prob > 0.3 and max_prob < 0.7) or predicted_class == 'Neutral':
                prediction = 'Neutral'
            else:
                prediction = predicted_class
        else:
            prediction = predicted_class
        
        # Get category
        category = assign_category(review_text)
        
        return jsonify({
            'sentiment': prediction,
            'probabilities': prob_dict,
            'category': category,
            'review': review_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predict sentiment for multiple reviews"""
    if not model_trained:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    data = request.json
    reviews = data.get('reviews', [])
    
    if not reviews:
        return jsonify({'error': 'Reviews list is required'}), 400
    
    try:
        results = []
        classes = model.classes_
        for review in reviews:
            review_tfidf = tfidf_vectorizer.transform([review])
            probabilities = model.predict_proba(review_tfidf)[0]
            
            # Ensure probabilities is a numpy array (not sparse)
            if hasattr(probabilities, 'toarray'):
                probabilities = probabilities.toarray()[0]
            probabilities = np.array(probabilities).flatten()
            
            # Improved prediction: Use probability threshold for neutral
            max_prob = float(np.max(probabilities))
            max_idx = int(np.argmax(probabilities))
            predicted_class = classes[max_idx]
            
            # Check if neutral is in classes and has reasonable probability
            if 'Neutral' in classes:
                neutral_idx = list(classes).index('Neutral')
                neutral_prob = probabilities[neutral_idx]
                
                # If neutral probability is high (>0.3) and max prob is not very confident (<0.7)
                # or if neutral is the highest, use neutral
                if (neutral_prob > 0.3 and max_prob < 0.7) or predicted_class == 'Neutral':
                    prediction = 'Neutral'
                else:
                    prediction = predicted_class
            else:
                prediction = predicted_class
            
            category = assign_category(review)
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            
            results.append({
                'review': review,
                'sentiment': prediction,
                'probabilities': prob_dict,
                'category': category
            })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Check required columns
        if 'Review' not in df.columns:
            return jsonify({'error': 'CSV must contain a "Review" column'}), 400
        
        # Preprocess reviews
        df['cleaned_review'] = df['Review'].apply(preprocess_text)
        
        # Predict sentiments if model is trained
        if model_trained:
            reviews_list = df['Review'].tolist()
            review_tfidf = tfidf_vectorizer.transform(reviews_list)
            probabilities_all = model.predict_proba(review_tfidf)
            classes = model.classes_
            
            # Improved predictions with neutral threshold
            predictions = []
            for probs in probabilities_all:
                # Ensure probs is a numpy array (not sparse)
                if hasattr(probs, 'toarray'):
                    probs = probs.toarray()[0]
                probs = np.array(probs).flatten()
                
                max_prob = float(np.max(probs))
                max_idx = int(np.argmax(probs))
                predicted_class = classes[max_idx]
                
                # Check if neutral is in classes and has reasonable probability
                if 'Neutral' in classes:
                    neutral_idx = list(classes).index('Neutral')
                    neutral_prob = probs[neutral_idx]
                    
                    # If neutral probability is high (>0.3) and max prob is not very confident (<0.7)
                    # or if neutral is the highest, use neutral
                    if (neutral_prob > 0.3 and max_prob < 0.7) or predicted_class == 'Neutral':
                        predictions.append('Neutral')
                    else:
                        predictions.append(predicted_class)
                else:
                    predictions.append(predicted_class)
            
            df['Predicted_Sentiment'] = predictions
            df['Category'] = df['Review'].apply(assign_category)
            
            # Add probabilities for each class
            probabilities = probabilities_all
            
            # Add probabilities
            for i, cls in enumerate(classes):
                df[f'Probability_{cls}'] = probabilities_all[:, i]
        
        # Save processed file
        output_filename = f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_filename, index=False)
        
        # Calculate statistics
        stats = {
            'total_reviews': len(df),
            'file_name': output_filename
        }
        
        if 'Predicted_Sentiment' in df.columns:
            stats['sentiment_distribution'] = df['Predicted_Sentiment'].value_counts().to_dict()
            stats['category_distribution'] = df['Category'].value_counts().to_dict()
        
        return jsonify({
            'message': 'File processed successfully',
            'stats': stats,
            'filename': output_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train or retrain the model"""
    global model, tfidf_vectorizer, model_accuracy, model_trained
    
    data = request.json
    use_cleaned = data.get('use_cleaned', True)
    test_size = data.get('test_size', 0.2)
    
    try:
        # Load data
        filename = 'reviews_cleaned.csv' if use_cleaned and os.path.exists('reviews_cleaned.csv') else 'reviews_with_sentiment.csv'
        
        if not os.path.exists(filename):
            return jsonify({'error': f'Data file {filename} not found'}), 404
        
        df = pd.read_csv(filename)
        df = df.dropna(subset=['Review', 'Sentiment'])
        
        # Use cleaned review if available, otherwise use original
        if 'cleaned_review' in df.columns and use_cleaned:
            X = df['cleaned_review'].fillna('')
        else:
            X = df['Review'].fillna('')
        
        y = df['Sentiment']
        
        # Vectorize
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(X)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=test_size, random_state=42
        )
        
        # Use class_weight='balanced' to handle imbalanced classes (especially Neutral)
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracy = accuracy
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_list = cm.tolist()
        
        # Save model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        model_info = {
            'accuracy': float(accuracy),
            'trained_date': datetime.now().isoformat(),
            'test_size': test_size,
            'training_samples': X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train),
            'test_samples': X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
        }
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f)
        
        model_trained = True
        
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm_list,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the dataset"""
    try:
        stats = {}
        
        # Try to load different data files
        files_to_check = [
            ('reviews_with_sentiment.csv', 'sentiment'),
            ('reviews_cleaned.csv', 'cleaned'),
            ('reviews_with_categories.csv', 'categories')
        ]
        
        for filename, key in files_to_check:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                
                # Base statistics
                dataset_stats = {
                    'total_reviews': len(df),
                    'columns': list(df.columns),
                    'file_name': filename
                }
                
                # Common statistics
                if 'Sentiment' in df.columns:
                    dataset_stats['sentiment_distribution'] = df['Sentiment'].value_counts().to_dict()
                
                if 'Rating' in df.columns:
                    dataset_stats['rating_distribution'] = df['Rating'].value_counts().to_dict()
                
                # Dataset-specific statistics
                if key == 'cleaned' and 'cleaned_review' in df.columns:
                    # Show cleaned vs original comparison
                    original_lengths = df['Review'].astype(str).str.len()
                    cleaned_lengths = df['cleaned_review'].astype(str).str.len()
                    dataset_stats['preprocessing_stats'] = {
                        'avg_original_length': float(original_lengths.mean()),
                        'avg_cleaned_length': float(cleaned_lengths.mean()),
                        'reduction_percentage': float(((original_lengths - cleaned_lengths) / original_lengths * 100).mean())
                    }
                    # Show empty cleaned reviews
                    empty_cleaned = (df['cleaned_review'].astype(str).str.strip() == '').sum()
                    dataset_stats['empty_cleaned_reviews'] = int(empty_cleaned)
                
                if key == 'categories':
                    # Category distribution
                    cat_col = 'Complaint_Category' if 'Complaint_Category' in df.columns else 'Category'
                    if cat_col in df.columns:
                        dataset_stats['category_distribution'] = df[cat_col].value_counts().to_dict()
                        dataset_stats['total_categories'] = df[cat_col].nunique()
                
                if key == 'sentiment':
                    # Show sentiment distribution details
                    if 'Sentiment' in df.columns:
                        dataset_stats['sentiment_percentages'] = {
                            sentiment: f"{(count / len(df) * 100):.2f}%"
                            for sentiment, count in df['Sentiment'].value_counts().items()
                        }
                
                stats[key] = dataset_stats
        
        # Model info
        if os.path.exists('model_info.json'):
            with open('model_info.json', 'r') as f:
                stats['model'] = json.load(f)
        else:
            stats['model'] = {
                'trained': model_trained,
                'accuracy': model_accuracy
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed files"""
    try:
        return send_from_directory('.', filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess uploaded data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        df = pd.read_csv(file)
        
        if 'Review' not in df.columns:
            return jsonify({'error': 'CSV must contain a "Review" column'}), 400
        
        # Preprocess
        df['cleaned_review'] = df['Review'].apply(preprocess_text)
        
        # Add sentiment if rating exists
        if 'Rating' in df.columns:
            df['Sentiment'] = df['Rating'].apply(
                lambda x: 'Negative' if x < 3 else ('Positive' if x > 3 else 'Neutral')
            )
        
        # Add categories
        df['Category'] = df['Review'].apply(assign_category)
        
        # Save
        output_filename = f'preprocessed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_filename, index=False)
        
        return jsonify({
            'message': 'Data preprocessed successfully',
            'filename': output_filename,
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/compare', methods=['POST'])
def compare_models():
    """Compare multiple ML models"""
    try:
        data = request.json
        use_cleaned = data.get('use_cleaned', True)
        cv_folds = data.get('cv_folds', 5)
        use_cache = data.get('use_cache', True)
        force_retrain = data.get('force_retrain', False)
        
        # Load data
        filename = 'reviews_cleaned.csv' if use_cleaned and os.path.exists('reviews_cleaned.csv') else 'reviews_with_sentiment.csv'
        if not os.path.exists(filename):
            return jsonify({'error': 'Data file not found'}), 404
        
        df = pd.read_csv(filename)
        df = df.dropna(subset=['Review', 'Sentiment'])
        
        # Prepare data
        if 'cleaned_review' in df.columns and use_cleaned:
            X = df['cleaned_review'].fillna('')
        else:
            X = df['Review'].fillna('')
        
        y = df['Sentiment']
        
        # Vectorize
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
        
        # Train all models (with caching)
        results = model_manager.train_all_models(X_train, y_train, X_test, y_test, cv_folds, use_cache, force_retrain)
        
        # Save to cache if not from cache
        if use_cache and not any(r.get('from_cache', False) for r in results.values()):
            # Generate cache key using the same method as ModelManager
            import hashlib
            data_hash = hashlib.md5(
                f"{X_train.shape[0]}_{X_train.shape[1]}_{y_train.shape[0]}_{cv_folds}".encode()
            ).hexdigest()[:12]
            cache_key = f"comparison_{data_hash}"
            model_manager.save_comparison_cache(cache_key, results, tfidf)
        
        # Format response
        comparison = {}
        cache_info = {}
        for name, result in results.items():
            comparison[name] = result['metrics']
            cache_info[name] = result.get('from_cache', False)
        
        return jsonify({
            'comparison': comparison,
            'best_model': max(comparison.keys(), key=lambda k: comparison[k]['f1_score']),
            'total_models': len(comparison),
            'from_cache': any(cache_info.values()),
            'cache_info': cache_info,
            'message': 'Loaded from cache' if any(cache_info.values()) else 'Models trained fresh'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/tune', methods=['POST'])
def tune_hyperparameters():
    """Perform hyperparameter tuning"""
    try:
        data = request.json
        model_name = data.get('model_name')
        use_cleaned = data.get('use_cleaned', True)
        
        if model_name not in model_manager.models:
            return jsonify({'error': f'Model {model_name} not supported'}), 400
        
        # Load data
        filename = 'reviews_cleaned.csv' if use_cleaned and os.path.exists('reviews_cleaned.csv') else 'reviews_with_sentiment.csv'
        if not os.path.exists(filename):
            return jsonify({'error': 'Data file not found'}), 404
        
        df = pd.read_csv(filename)
        df = df.dropna(subset=['Review', 'Sentiment'])
        
        # Prepare data
        if 'cleaned_review' in df.columns and use_cleaned:
            X = df['cleaned_review'].fillna('')
        else:
            X = df['Review'].fillna('')
        
        y = df['Sentiment']
        
        # Vectorize
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(X)
        
        # Tune hyperparameters
        param_grid = PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            return jsonify({'error': f'No parameter grid for {model_name}'}), 400
        
        result = model_manager.hyperparameter_tuning(model_name, X_tfidf, y, param_grid)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nlp/topics', methods=['POST'])
def extract_topics():
    """Extract topics from text"""
    try:
        data = request.json
        texts = data.get('texts', [])
        n_topics = data.get('n_topics', 5)
        method = data.get('method', 'lda')
        
        if not texts:
            return jsonify({'error': 'Texts list is required', 'topics': []}), 400
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter empty texts
        texts = [t for t in texts if t and str(t).strip()]
        
        if not texts:
            return jsonify({'error': 'No valid texts provided', 'topics': []}), 400
        
        topics = nlp_processor.topic_modeling(texts, n_topics, method)
        return jsonify({'topics': topics if topics else []})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'topics': []}), 500

@app.route('/api/nlp/keywords', methods=['POST'])
def extract_keywords():
    """Extract keywords from text"""
    try:
        data = request.json
        texts = data.get('texts', [])
        n_keywords = data.get('n_keywords', 20)
        
        if not texts:
            return jsonify({'error': 'Texts list is required', 'keywords': []}), 400
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter empty texts
        texts = [t for t in texts if t and str(t).strip()]
        
        if not texts:
            return jsonify({'error': 'No valid texts provided', 'keywords': []}), 400
        
        keywords = nlp_processor.extract_keywords(texts, n_keywords)
        return jsonify({'keywords': keywords if keywords else []})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'keywords': []}), 500

@app.route('/api/nlp/aspects', methods=['POST'])
def extract_aspects():
    """Extract aspects from text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text or not str(text).strip():
            return jsonify({'error': 'Text is required', 'aspects': []}), 400
        
        aspects = nlp_processor.extract_aspects(str(text))
        return jsonify({'aspects': aspects if aspects else []})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'aspects': []}), 500

@app.route('/api/nlp/emotions', methods=['POST'])
def extract_emotions():
    """Extract emotions from text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text or not str(text).strip():
            return jsonify({'error': 'Text is required', 'emotions': {}, 'dominant_emotion': None}), 400
        
        emotions = nlp_processor.extract_emotions(str(text))
        return jsonify(emotions if emotions else {'emotions': {}, 'dominant_emotion': None})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'emotions': {}, 'dominant_emotion': None}), 500

@app.route('/api/interpret/explain', methods=['POST'])
def explain_prediction():
    """Explain a prediction"""
    try:
        if not model_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        data = request.json
        text = data.get('text', '')
        top_features = data.get('top_features', 10)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        interpreter = ModelInterpreter(model, tfidf_vectorizer)
        explanation = interpreter.explain_prediction(text, top_features)
        
        return jsonify(explanation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/interpret/features', methods=['GET'])
def get_feature_importance():
    """Get feature importance"""
    try:
        if not model_trained:
            return jsonify({'error': 'Model not trained yet', 'feature_importance': []}), 400
        
        # Load training data
        filename = 'reviews_cleaned.csv' if os.path.exists('reviews_cleaned.csv') else 'reviews_with_sentiment.csv'
        if not os.path.exists(filename):
            return jsonify({'error': 'Training data not found', 'feature_importance': []}), 404
        
        df = pd.read_csv(filename)
        df = df.dropna(subset=['Review', 'Sentiment'])
        
        if len(df) == 0:
            return jsonify({'error': 'No training data available', 'feature_importance': []}), 404
        
        X = df['Review'].fillna('')
        y = df['Sentiment']
        
        X_tfidf = tfidf_vectorizer.transform(X)
        
        interpreter = ModelInterpreter(model, tfidf_vectorizer)
        importance = interpreter.get_feature_importance(X_tfidf, y, n_features=30)
        
        if not importance or len(importance) == 0:
            return jsonify({'error': 'Could not calculate feature importance', 'feature_importance': []}), 500
        
        return jsonify({'feature_importance': importance})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'feature_importance': []}), 500

@app.route('/api/timeseries/trends', methods=['POST'])
def get_sentiment_trends():
    """Get sentiment trends over time"""
    try:
        data = request.json
        date_column = data.get('date_column', 'Date')
        sentiment_column = data.get('sentiment_column', 'Sentiment')
        
        # Load data
        filename = data.get('filename', 'reviews_with_sentiment.csv')
        if not os.path.exists(filename):
            return jsonify({'error': 'Data file not found'}), 404
        
        df = pd.read_csv(filename)
        trends = time_analyzer.analyze_sentiment_trends(df, date_column, sentiment_column)
        
        return jsonify(trends)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/timeseries/shifts', methods=['POST'])
def detect_shifts():
    """Detect sentiment shifts"""
    try:
        data = request.json
        filename = data.get('filename', 'reviews_with_sentiment.csv')
        if not os.path.exists(filename):
            return jsonify({'error': 'Data file not found'}), 404
        
        df = pd.read_csv(filename)
        shifts = time_analyzer.detect_sentiment_shifts(df)
        
        return jsonify({'shifts': shifts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/report', methods=['POST'])
def export_report():
    """Export comprehensive report"""
    try:
        data = request.json
        format_type = data.get('format', 'json')
        
        # Get model info
        model_info = {
            'accuracy': model_accuracy,
            'model_trained': model_trained
        }
        
        # Get statistics
        stats = db_manager.get_statistics()
        
        # Generate report
        if format_type == 'json':
            filename = export_manager.generate_report_json({}, model_info, stats)
        elif format_type == 'excel':
            # Create summary
            filename = export_manager.create_summary_report(
                pd.read_csv('reviews_with_sentiment.csv') if os.path.exists('reviews_with_sentiment.csv') else pd.DataFrame(),
                model_info
            )
            return jsonify({'files': filename})
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        return jsonify({'filename': filename, 'message': 'Report exported successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/models', methods=['GET'])
def get_models_from_db():
    """Get all models from database"""
    try:
        models = db_manager.get_active_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/predictions', methods=['GET'])
def get_predictions_from_db():
    """Get recent predictions from database"""
    try:
        limit = request.args.get('limit', 100, type=int)
        predictions = db_manager.get_recent_predictions(limit)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize airline insights
airline_scraper = AirlineReviewScraper()
airline_analyzer = AirlineInsightsAnalyzer(nlp_processor)

@app.route('/api/airline/scrape', methods=['POST'])
def scrape_airline_reviews():
    """Scrape live reviews for an airline"""
    try:
        data = request.json
        airline_name = data.get('airline_name', '')
        custom_url = data.get('url', '')
        max_pages = data.get('max_pages', 5)
        
        if custom_url:
            df = airline_scraper.scrape_custom_url(custom_url, max_pages)
        elif airline_name:
            df = airline_scraper.scrape_trustpilot_reviews(airline_name, max_pages)
        else:
            return jsonify({'error': 'Either airline_name or url is required'}), 400
        
        if df.empty:
            return jsonify({'error': 'No reviews found', 'reviews': []}), 404
        
        # Save scraped reviews
        filename = f'airline_reviews_{airline_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        
        return jsonify({
            'message': f'Successfully scraped {len(df)} reviews',
            'total_reviews': len(df),
            'filename': filename,
            'reviews': df.to_dict('records')[:50]  # Return first 50
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/airline/analyze', methods=['POST'])
def analyze_airline():
    """Analyze airline reviews and generate insights"""
    try:
        data = request.json
        airline_name = data.get('airline_name', 'Unknown')
        reviews = data.get('reviews', [])
        filename = data.get('filename', '')
        
        # Load reviews if filename provided
        if filename and os.path.exists(filename):
            df = pd.read_csv(filename)
        elif reviews:
            df = pd.DataFrame(reviews)
            if 'Review' not in df.columns:
                return jsonify({'error': 'Reviews must contain Review column'}), 400
        else:
            return jsonify({'error': 'Either reviews or filename required'}), 400
        
        # Generate insights
        insights = airline_analyzer.analyze_airline_reviews(df, airline_name)
        
        # Add NLP analysis if available
        if nlp_processor and len(df) > 0:
            try:
                # Extract aspects from negative reviews
                sentiment_col = 'Sentiment' if 'Sentiment' in df.columns else None
                if not sentiment_col:
                    df['Sentiment'] = df['Rating'].apply(lambda x: 'Negative' if x <= 2 else 'Positive' if x >= 4 else 'Neutral')
                
                negative_reviews = df[df['Sentiment'] == 'Negative']['Review'].tolist()
                if negative_reviews:
                    aspects = nlp_processor.extract_aspects(negative_reviews[:20])
                    insights['key_aspects'] = aspects[:10]
            except Exception as e:
                print(f"NLP analysis error: {e}")
        
        return jsonify(insights)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/airline/scrape-and-analyze', methods=['POST'])
def scrape_and_analyze():
    """Scrape and analyze airline reviews in one go"""
    try:
        data = request.json
        airline_name = data.get('airline_name', '')
        custom_url = data.get('url', '')
        max_pages = data.get('max_pages', 5)
        
        # Scrape reviews
        if custom_url:
            df = airline_scraper.scrape_custom_url(custom_url, max_pages)
            airline_name = airline_name or 'Custom Airline'
        elif airline_name:
            df = airline_scraper.scrape_trustpilot_reviews(airline_name, max_pages)
        else:
            return jsonify({'error': 'Either airline_name or url is required'}), 400
        
        if df.empty:
            return jsonify({'error': 'No reviews found'}), 404
        
        # Analyze
        insights = airline_analyzer.analyze_airline_reviews(df, airline_name)
        
        # Add NLP analysis
        if nlp_processor and len(df) > 0:
            try:
                df['Sentiment'] = df['Rating'].apply(lambda x: 'Negative' if x <= 2 else 'Positive' if x >= 4 else 'Neutral')
                negative_reviews = df[df['Sentiment'] == 'Negative']['Review'].tolist()
                if negative_reviews:
                    aspects = nlp_processor.extract_aspects(negative_reviews[:20])
                    insights['key_aspects'] = aspects[:10]
            except:
                pass
        
        # Save
        filename = f'airline_reviews_{airline_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        insights['filename'] = filename
        
        return jsonify(insights)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve React app"""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

