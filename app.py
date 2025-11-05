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

POSITIVE_INDICATORS = [
    'excellent', 'outstanding', 'amazing', 'wonderful', 'fantastic', 'perfect', 'flawless',
    'impressed', 'love', 'loved', 'great', 'best', 'awesome', 'brilliant', 'superb',
    'exceptional', 'delighted', 'pleased', 'satisfied', 'thrilled', 'happy', 'glad',
    'recommend', 'highly recommend', 'exceeded expectations', 'above and beyond',
    'top notch', 'premium', 'luxury', 'outstanding service', 'stellar', 'phenomenal'
]

NEGATIVE_INDICATORS = [
    'terrible', 'awful', 'horrible', 'worst', 'disappointed', 'disappointing', 'disgusting',
    'hate', 'hated', 'poor', 'bad', 'very bad', 'pathetic', 'ridiculous', 'unacceptable',
    'frustrated', 'angry', 'upset', 'complaint', 'complain', 'refund', 'scam', 'fraud',
    'waste', 'waste of money', 'never again', 'avoid', 'warning', 'beware', 'disaster'
]

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

def get_sentiment_boost(text):
    """
    Analyze text for strong positive/negative indicators to boost confidence
    Returns: 'positive' if strong positive indicators found, 'negative' if negative, None otherwise
    """
    if not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Count positive and negative indicators
    positive_count = sum(1 for indicator in POSITIVE_INDICATORS if indicator in text_lower)
    negative_count = sum(1 for indicator in NEGATIVE_INDICATORS if indicator in text_lower)
    
    # Strong signal if 3+ indicators found
    if positive_count >= 3 and negative_count == 0:
        return 'positive'
    elif negative_count >= 3 and positive_count == 0:
        return 'negative'
    elif positive_count >= 2 and positive_count > negative_count * 2:
        return 'positive'
    elif negative_count >= 2 and negative_count > positive_count * 2:
        return 'negative'
    
    return None

def is_valid_review(text):
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False
    
    text_lower = text.lower().strip()
    text_no_space = text_lower.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    
    if len(text_no_space) < 10:
        return False
    
    unique_chars = set(text_no_space)
    if len(unique_chars) < 5:
        return False
    
    consonant_count = sum(1 for c in text_no_space if c.isalpha() and c not in 'aeiou')
    vowel_count = sum(1 for c in text_no_space if c in 'aeiou')
    
    if len(text_no_space) > 8 and vowel_count == 0:
        return False
    
    if len(text_no_space) > 8 and consonant_count > 0:
        if vowel_count == 0 or (vowel_count / max(consonant_count, 1)) < 0.15:
            return False
    
    repeated_chars = sum(1 for i in range(len(text_no_space)-2) if text_no_space[i] == text_no_space[i+1] == text_no_space[i+2])
    if repeated_chars > len(text_no_space) * 0.12:
        return False
    
    words = text_lower.split()
    if len(words) < 3:
        return False
    
    if len(words) == 1 and len(words[0]) > 8:
        avg_word_len = len(words[0])
        if avg_word_len > 10 and vowel_count == 0:
            return False
    
    airline_keywords = [
        'flight', 'airline', 'airplane', 'airport', 'booking', 'ticket', 'check-in', 'checkin',
        'passenger', 'cabin', 'crew', 'pilot', 'steward', 'attendant', 'luggage', 'baggage',
        'delay', 'departure', 'arrival', 'gate', 'boarding', 'seat', 'meal', 'service',
        'aircraft', 'takeoff', 'landing', 'terminal', 'security', 'customs', 'immigration'
    ]
    
    review_indicators = [
        'experience', 'review', 'recommend', 'satisfied', 'disappointed', 'excellent',
        'poor', 'good', 'bad', 'great', 'terrible', 'amazing', 'awful', 'wonderful',
        'horrible', 'fantastic', 'worst', 'best', 'love', 'hate', 'enjoy', 'dislike'
    ]
    
    has_airline_keyword = any(keyword in text_lower for keyword in airline_keywords)
    has_review_indicator = any(indicator in text_lower for indicator in review_indicators)
    
    if has_airline_keyword or has_review_indicator:
        return True
    
    if len(words) >= 15:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 2.5 <= avg_word_len <= 8.0:
            return True
    
    return False


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
    
    if not is_valid_review(review_text):
        return jsonify({'error': 'This is not a valid airline review. Please provide a meaningful review about your flight experience.'}), 400
    
    try:
        # Preprocess text before prediction (critical: model was trained on preprocessed text)
        preprocessed_text = preprocess_text(review_text)
        
        # Transform preprocessed text
        review_tfidf = tfidf_vectorizer.transform([preprocessed_text])
        probabilities = model.predict_proba(review_tfidf)[0]
        classes = model.classes_
        
        # Ensure probabilities is a numpy array (not sparse)
        if hasattr(probabilities, 'toarray'):
            probabilities = probabilities.toarray()[0]
        probabilities = np.array(probabilities).flatten()
        
        # Create probability dict
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        # Prediction logic: Trust the model when it's confident
        max_prob = float(np.max(probabilities))
        max_idx = int(np.argmax(probabilities))
        predicted_class = classes[max_idx]
        
        sentiment_boost = get_sentiment_boost(review_text)
        if 'Neutral' in classes:
            neutral_idx = list(classes).index('Neutral')
            neutral_prob = probabilities[neutral_idx]
            
            sorted_indices = np.argsort(probabilities)[::-1]
            second_highest_prob = float(probabilities[sorted_indices[1]]) if len(sorted_indices) > 1 else 0
            
            if max_prob > 0.80:
                prediction = predicted_class
            elif max_prob > 0.35 and (max_prob - second_highest_prob) < 0.05 and sentiment_boost:
                if sentiment_boost == 'positive' and 'Positive' in classes:
                    prediction = 'Positive'
                elif sentiment_boost == 'negative' and 'Negative' in classes:
                    prediction = 'Negative'
                else:
                    prediction = predicted_class
            elif max_prob < 0.55 and neutral_prob > 0.35 and predicted_class != 'Neutral':
                if neutral_prob > (max_prob - 0.1):
                    prediction = 'Neutral'
                else:
                    prediction = predicted_class
            else:
                prediction = predicted_class
        else:
            prediction = predicted_class
        
        # Get category
        category = assign_category(review_text)
        
        # Calculate dynamic feature importance for this specific prediction
        feature_importance = []
        try:
            if hasattr(model, 'coef_'):
                # Direct calculation using coefficients and TF-IDF values
                feature_names = tfidf_vectorizer.get_feature_names_out()
                text_features = review_tfidf.toarray()[0]
                non_zero_indices = np.nonzero(text_features)[0]
                
                if len(non_zero_indices) > 0:
                    predicted_class_idx = list(classes).index(prediction)
                    coefs = model.coef_[predicted_class_idx]
                    
                    feature_scores = []
                    for idx in non_zero_indices:
                        score = float(coefs[idx]) * float(text_features[idx])
                        if abs(score) > 0.0001:  # Only include significant contributions
                            feature_scores.append({
                                'feature': str(feature_names[idx]),
                                'importance': score
                            })
                    
                    if len(feature_scores) > 0:
                        feature_scores.sort(key=lambda x: abs(x['importance']), reverse=True)
                        feature_importance = feature_scores[:15]
                        print(f"Calculated {len(feature_importance)} feature importance items")
                    else:
                        print("No significant feature contributions found")
                else:
                    print("No non-zero features found in text")
            else:
                # Try using ModelInterpreter as fallback
                interpreter = ModelInterpreter(model, tfidf_vectorizer)
                explanation = interpreter.explain_prediction(preprocessed_text, top_features=15)
                if explanation and 'top_features' in explanation and len(explanation['top_features']) > 0:
                    feature_importance = [
                        {
                            'feature': f['feature'],
                            'importance': f['contribution']
                        }
                        for f in explanation['top_features']
                    ]
        except Exception as e:
            import traceback
            print(f"Error calculating feature importance: {e}")
            traceback.print_exc()
        
        response_data = {
            'sentiment': prediction,
            'probabilities': prob_dict,
            'category': category,
            'review': review_text
        }
        
        if len(feature_importance) > 0:
            response_data['feature_importance'] = feature_importance
            print(f"Returning {len(feature_importance)} feature importance items in response")
        else:
            print("Warning: Feature importance is empty, not including in response")
        
        return jsonify(response_data)
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
            if not is_valid_review(review):
                results.append({
                    'review': review,
                    'sentiment': None,
                    'error': 'Not a valid airline review',
                    'probabilities': {},
                    'category': None
                })
                continue
            
            # Preprocess text before prediction (critical: model was trained on preprocessed text)
            preprocessed_text = preprocess_text(review)
            
            # Transform preprocessed text
            review_tfidf = tfidf_vectorizer.transform([preprocessed_text])
            probabilities = model.predict_proba(review_tfidf)[0]
            
            # Ensure probabilities is a numpy array (not sparse)
            if hasattr(probabilities, 'toarray'):
                probabilities = probabilities.toarray()[0]
            probabilities = np.array(probabilities).flatten()
            
            # Prediction logic: Trust the model when it's confident
            max_prob = float(np.max(probabilities))
            max_idx = int(np.argmax(probabilities))
            predicted_class = classes[max_idx]
            
            sentiment_boost = get_sentiment_boost(review)
            if 'Neutral' in classes:
                neutral_idx = list(classes).index('Neutral')
                neutral_prob = probabilities[neutral_idx]
                
                sorted_indices = np.argsort(probabilities)[::-1]
                second_highest_prob = float(probabilities[sorted_indices[1]]) if len(sorted_indices) > 1 else 0
                
                if max_prob > 0.80:
                    prediction = predicted_class
                elif max_prob > 0.35 and (max_prob - second_highest_prob) < 0.05 and sentiment_boost:
                    if sentiment_boost == 'positive' and 'Positive' in classes:
                        prediction = 'Positive'
                    elif sentiment_boost == 'negative' and 'Negative' in classes:
                        prediction = 'Negative'
                    else:
                        prediction = predicted_class
                elif max_prob < 0.55 and neutral_prob > 0.35 and predicted_class != 'Neutral':
                    if neutral_prob > (max_prob - 0.1):
                        prediction = 'Neutral'
                    else:
                        prediction = predicted_class
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
        df['is_valid'] = df['Review'].apply(lambda x: is_valid_review(str(x)))
        
        # Predict sentiments if model is trained
        if model_trained:
            # Filter valid reviews for prediction
            valid_df = df[df['is_valid']].copy()
            
            if len(valid_df) == 0:
                return jsonify({'error': 'No valid airline reviews found in the file'}), 400
            
            # Use preprocessed reviews for prediction (critical: model was trained on preprocessed text)
            cleaned_reviews_list = valid_df['cleaned_review'].tolist()
            review_tfidf = tfidf_vectorizer.transform(cleaned_reviews_list)
            probabilities_all = model.predict_proba(review_tfidf)
            classes = model.classes_
            
            # Improved predictions with better neutral threshold logic
            predictions = []
            valid_idx = 0
            for idx in range(len(df)):
                if not df.iloc[idx]['is_valid']:
                    predictions.append('Invalid Review')
                    continue
                
                probs = probabilities_all[valid_idx]
                valid_idx += 1
                
                # Ensure probs is a numpy array (not sparse)
                if hasattr(probs, 'toarray'):
                    probs = probs.toarray()[0]
                probs = np.array(probs).flatten()
                
                max_prob = float(np.max(probs))
                max_idx = int(np.argmax(probs))
                predicted_class = classes[max_idx]
                
                original_review = df['Review'].iloc[idx] if 'Review' in df.columns else ''
                sentiment_boost = get_sentiment_boost(original_review)
                
                if 'Neutral' in classes:
                    neutral_idx = list(classes).index('Neutral')
                    neutral_prob = probs[neutral_idx]
                    
                    sorted_indices = np.argsort(probs)[::-1]
                    second_highest_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0
                    
                    if max_prob > 0.80:
                        predictions.append(predicted_class)
                    elif max_prob > 0.35 and (max_prob - second_highest_prob) < 0.05 and sentiment_boost:
                        if sentiment_boost == 'positive' and 'Positive' in classes:
                            predictions.append('Positive')
                        elif sentiment_boost == 'negative' and 'Negative' in classes:
                            predictions.append('Negative')
                        else:
                            predictions.append(predicted_class)
                    elif max_prob < 0.55 and neutral_prob > 0.35 and predicted_class != 'Neutral':
                        if neutral_prob > (max_prob - 0.1):
                            predictions.append('Neutral')
                        else:
                            predictions.append(predicted_class)
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

@app.route('/api/airline/compare', methods=['POST'])
def compare_airlines():
    """Compare multiple airlines using the trained model"""
    try:
        if not model_trained:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
        
        data = request.json
        airline_names = data.get('airline_names', [])
        max_pages = data.get('max_pages', 3)
        
        if not airline_names or len(airline_names) < 2:
            return jsonify({'error': 'Please provide at least 2 airlines to compare'}), 400
        
        comparison_results = {}
        all_airlines_data = []
        
        # Scrape and analyze each airline
        for airline_name in airline_names:
            try:
                # Scrape reviews
                df = airline_scraper.scrape_trustpilot_reviews(airline_name, max_pages)
                
                if df.empty:
                    comparison_results[airline_name] = {
                        'error': 'No reviews found',
                        'total_reviews': 0
                    }
                    continue
                
                # Use model to predict sentiment for all reviews
                valid_reviews = []
                predictions = []
                probabilities_list = []
                
                for review in df['Review'].tolist():
                    if not is_valid_review(str(review)):
                        continue
                    
                    try:
                        # Preprocess and predict
                        preprocessed_text = preprocess_text(str(review))
                        review_tfidf = tfidf_vectorizer.transform([preprocessed_text])
                        probs = model.predict_proba(review_tfidf)[0]
                        
                        # Ensure probs is numpy array
                        if hasattr(probs, 'toarray'):
                            probs = probs.toarray()[0]
                        probs = np.array(probs).flatten()
                        
                        # Get prediction
                        max_idx = int(np.argmax(probs))
                        predicted_class = model.classes_[max_idx]
                        
                        # Apply prediction logic
                        sentiment_boost = get_sentiment_boost(str(review))
                        
                        if 'Neutral' in model.classes_:
                            neutral_idx = list(model.classes_).index('Neutral')
                            neutral_prob = probs[neutral_idx]
                            max_prob = float(np.max(probs))
                            
                            sorted_indices = np.argsort(probs)[::-1]
                            second_highest_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0
                            
                            if max_prob > 0.80:
                                final_prediction = predicted_class
                            elif max_prob > 0.35 and (max_prob - second_highest_prob) < 0.05 and sentiment_boost:
                                if sentiment_boost == 'positive' and 'Positive' in model.classes_:
                                    final_prediction = 'Positive'
                                elif sentiment_boost == 'negative' and 'Negative' in model.classes_:
                                    final_prediction = 'Negative'
                                else:
                                    final_prediction = predicted_class
                            elif max_prob < 0.55 and neutral_prob > 0.35 and predicted_class != 'Neutral':
                                if neutral_prob > (max_prob - 0.1):
                                    final_prediction = 'Neutral'
                                else:
                                    final_prediction = predicted_class
                            else:
                                final_prediction = predicted_class
                        else:
                            final_prediction = predicted_class
                        
                        valid_reviews.append(review)
                        predictions.append(final_prediction)
                        probabilities_list.append({cls: float(prob) for cls, prob in zip(model.classes_, probs)})
                        
                    except Exception as e:
                        print(f"Error predicting for review: {e}")
                        continue
                
                if len(valid_reviews) == 0:
                    comparison_results[airline_name] = {
                        'error': 'No valid reviews found for sentiment analysis',
                        'total_reviews': len(df)
                    }
                    continue
                
                # Add predictions to dataframe
                df_predictions = df[df['Review'].isin(valid_reviews)].copy()
                df_predictions['Model_Predicted_Sentiment'] = predictions
                df_predictions['Model_Confidence'] = [max(probs.values()) for probs in probabilities_list]
                
                # Calculate statistics
                sentiment_dist = pd.Series(predictions).value_counts().to_dict()
                avg_confidence = np.mean([max(probs.values()) for probs in probabilities_list])
                avg_rating = df_predictions['Rating'].mean()
                
                # Get category breakdown
                df_predictions['Category'] = df_predictions['Review'].apply(assign_category)
                category_dist = df_predictions['Category'].value_counts().to_dict()
                
                # Analyze top issues using model predictions
                # Add 'Sentiment' column for _identify_issues method compatibility
                df_predictions['Sentiment'] = df_predictions['Model_Predicted_Sentiment']
                
                # Check for negative reviews using the Sentiment column
                negative_reviews_df = df_predictions[df_predictions['Sentiment'] == 'Negative']
                if len(negative_reviews_df) > 0:
                    try:
                        top_issues = airline_analyzer._identify_issues(df_predictions)
                    except Exception as e:
                        print(f"Error identifying issues: {e}")
                        import traceback
                        traceback.print_exc()
                        top_issues = {'top_5_issues': {}}
                else:
                    top_issues = {'top_5_issues': {}}
                
                # Store results
                comparison_results[airline_name] = {
                    'airline': airline_name,
                    'total_reviews': len(df),
                    'valid_reviews': len(valid_reviews),
                    'sentiment_distribution': sentiment_dist,
                    'model_confidence': float(avg_confidence),
                    'average_rating': float(avg_rating),
                    'category_distribution': category_dist,
                    'top_issues': top_issues.get('top_5_issues', {}),
                    'positive_percentage': (sentiment_dist.get('Positive', 0) / len(valid_reviews) * 100) if len(valid_reviews) > 0 else 0,
                    'negative_percentage': (sentiment_dist.get('Negative', 0) / len(valid_reviews) * 100) if len(valid_reviews) > 0 else 0,
                    'neutral_percentage': (sentiment_dist.get('Neutral', 0) / len(valid_reviews) * 100) if len(valid_reviews) > 0 else 0,
                }
                
                all_airlines_data.append({
                    'airline': airline_name,
                    'data': df_predictions
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                comparison_results[airline_name] = {
                    'error': str(e),
                    'total_reviews': 0
                }
        
        # Generate comparison summary
        comparison_summary = {
            'airlines_compared': len([r for r in comparison_results.values() if 'error' not in r]),
            'best_positive': None,
            'best_rating': None,
            'most_issues': None,
            'highest_confidence': None
        }
        
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        
        if valid_results:
            # Find best positive percentage
            best_positive = max(valid_results.items(), key=lambda x: x[1].get('positive_percentage', 0))
            comparison_summary['best_positive'] = {
                'airline': best_positive[0],
                'positive_percentage': best_positive[1]['positive_percentage']
            }
            
            # Find best rating
            best_rating = max(valid_results.items(), key=lambda x: x[1].get('average_rating', 0))
            comparison_summary['best_rating'] = {
                'airline': best_rating[0],
                'average_rating': best_rating[1]['average_rating']
            }
            
            # Find most issues (highest negative percentage)
            most_issues = max(valid_results.items(), key=lambda x: x[1].get('negative_percentage', 0))
            comparison_summary['most_issues'] = {
                'airline': most_issues[0],
                'negative_percentage': most_issues[1]['negative_percentage']
            }
            
            # Highest model confidence
            highest_conf = max(valid_results.items(), key=lambda x: x[1].get('model_confidence', 0))
            comparison_summary['highest_confidence'] = {
                'airline': highest_conf[0],
                'confidence': highest_conf[1]['model_confidence']
            }
        
        return jsonify({
            'comparison': comparison_results,
            'summary': comparison_summary,
            'total_airlines': len(airline_names)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve React app"""
    # Don't serve static files for API routes
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

