"""
Database Module for Model Versioning and Results Storage
Uses SQLite for lightweight database operations
"""

import sqlite3
import json
import os
from datetime import datetime
import pandas as pd

class DatabaseManager:
    """Manages database operations for model versioning and results"""
    
    def __init__(self, db_path='sentiment_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                accuracy REAL,
                f1_score REAL,
                precision REAL,
                recall REAL,
                training_samples INTEGER,
                test_samples INTEGER,
                parameters TEXT,
                model_file_path TEXT,
                vectorizer_file_path TEXT,
                trained_date TEXT,
                is_active INTEGER DEFAULT 0,
                notes TEXT
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                review_text TEXT,
                predicted_sentiment TEXT,
                confidence REAL,
                category TEXT,
                probabilities TEXT,
                created_at TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')
        
        # Training history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                training_date TEXT,
                accuracy REAL,
                f1_score REAL,
                training_time REAL,
                dataset_size INTEGER,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT,
                results TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_model(self, model_info):
        """Save model information to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO models (
                model_name, model_type, accuracy, f1_score, precision, recall,
                training_samples, test_samples, parameters, model_file_path,
                vectorizer_file_path, trained_date, is_active, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_info.get('model_name'),
            model_info.get('model_type'),
            model_info.get('accuracy'),
            model_info.get('f1_score'),
            model_info.get('precision'),
            model_info.get('recall'),
            model_info.get('training_samples'),
            model_info.get('test_samples'),
            json.dumps(model_info.get('parameters', {})),
            model_info.get('model_file_path'),
            model_info.get('vectorizer_file_path'),
            datetime.now().isoformat(),
            1,  # Set as active
            model_info.get('notes', '')
        ))
        
        model_id = cursor.lastrowid
        
        # Deactivate other models of the same type
        cursor.execute('''
            UPDATE models SET is_active = 0 
            WHERE model_type = ? AND id != ?
        ''', (model_info.get('model_type'), model_id))
        
        conn.commit()
        conn.close()
        return model_id
    
    def save_prediction(self, model_id, review_text, prediction_data):
        """Save a prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                model_id, review_text, predicted_sentiment, confidence,
                category, probabilities, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            review_text,
            prediction_data.get('sentiment'),
            prediction_data.get('confidence'),
            prediction_data.get('category'),
            json.dumps(prediction_data.get('probabilities', {})),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_models(self):
        """Get all active models"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM models WHERE is_active = 1 ORDER BY trained_date DESC
        ''', conn)
        conn.close()
        return df.to_dict('records')
    
    def get_model_history(self, model_name=None):
        """Get training history for models"""
        conn = sqlite3.connect(self.db_path)
        
        if model_name:
            query = '''
                SELECT * FROM models WHERE model_name = ? ORDER BY trained_date DESC
            '''
            df = pd.read_sql_query(query, conn, params=(model_name,))
        else:
            query = 'SELECT * FROM models ORDER BY trained_date DESC'
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df.to_dict('records')
    
    def get_recent_predictions(self, limit=100):
        """Get recent predictions"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT p.*, m.model_name 
            FROM predictions p
            JOIN models m ON p.model_id = m.id
            ORDER BY p.created_at DESC
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df.to_dict('records')
    
    def get_statistics(self):
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Total models
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM models')
        stats['total_models'] = cursor.fetchone()[0]
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # Prediction distribution
        cursor.execute('''
            SELECT predicted_sentiment, COUNT(*) as count 
            FROM predictions 
            GROUP BY predicted_sentiment
        ''')
        stats['sentiment_distribution'] = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        result = cursor.fetchone()[0]
        stats['avg_confidence'] = float(result) if result else 0.0
        
        conn.close()
        return stats

