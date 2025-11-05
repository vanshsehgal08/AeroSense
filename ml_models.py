"""
Advanced ML Models Module
Supports multiple algorithms with comparison and ensemble methods
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import os
import hashlib

class ModelManager:
    """Manages multiple ML models with comparison and selection"""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model = None
        self.cache_dir = 'models_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, X_train, y_train, cv_folds):
        """Generate cache key based on data and parameters"""
        # Create hash from data shape and sample of data
        data_hash = hashlib.md5(
            f"{X_train.shape[0]}_{X_train.shape[1]}_{y_train.shape[0]}_{cv_folds}".encode()
        ).hexdigest()[:12]
        return f"comparison_{data_hash}"
    
    def _load_cached_models(self, cache_key):
        """Load cached model comparison results"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}_results.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Load all models
                results = {}
                for name in self.models.keys():
                    model_file = os.path.join(self.cache_dir, f"{cache_key}_{name}.pkl")
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as mf:
                            model = pickle.load(mf)
                            results[name] = {
                                'model': model,
                                'metrics': cached_data.get(name, {}).get('metrics', {}),
                                'from_cache': True
                            }
                            self.trained_models[name] = model
                
                if len(results) == len(self.models):
                    return results, True
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        return None, False
    
    def _save_cached_models(self, cache_key, results, vectorizer):
        """Save model comparison results to cache"""
        try:
            # Save all models
            for name, result in results.items():
                model_file = os.path.join(self.cache_dir, f"{cache_key}_{name}.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(result['model'], f)
            
            # Save vectorizer
            vectorizer_file = os.path.join(self.cache_dir, f"{cache_key}_vectorizer.pkl")
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            # Save metrics
            cache_file = os.path.join(self.cache_dir, f"{cache_key}_results.json")
            cached_data = {
                name: {'metrics': result['metrics']} 
                for name, result in results.items()
            }
            cached_data['vectorizer_file'] = vectorizer_file
            cached_data['cached_date'] = datetime.now().isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
            
            print(f"Models cached with key: {cache_key}")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
    def train_all_models(self, X_train, y_train, X_test, y_test, cv_folds=5, use_cache=True, force_retrain=False):
        """Train all models and compare performance"""
        cache_key = self._get_cache_key(X_train, y_train, cv_folds)
        
        # Try to load from cache if enabled and not forcing retrain
        if use_cache and not force_retrain:
            cached_results, loaded = self._load_cached_models(cache_key)
            if loaded:
                print(f"Loaded {len(cached_results)} models from cache")
                # Recalculate metrics on test set for accuracy
                for name, result in cached_results.items():
                    model = result['model']
                    y_pred = model.predict(X_test)
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                    }
                    # Keep CV scores from cache if available
                    if 'metrics' in result and 'cv_mean' in result['metrics']:
                        metrics['cv_mean'] = result['metrics']['cv_mean']
                        metrics['cv_std'] = result['metrics']['cv_std']
                    else:
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                            metrics['cv_mean'] = float(cv_scores.mean())
                            metrics['cv_std'] = float(cv_scores.std())
                        except:
                            metrics['cv_mean'] = metrics['accuracy']
                            metrics['cv_std'] = 0.0
                    
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    metrics['classification_report'] = report
                    
                    cm = confusion_matrix(y_test, y_pred)
                    metrics['confusion_matrix'] = cm.tolist()
                    
                    cached_results[name]['metrics'] = metrics
                
                # Find best model
                best_model_name = max(cached_results.keys(), key=lambda k: cached_results[k]['metrics']['f1_score'])
                self.best_model = cached_results[best_model_name]['model']
                self.model_metrics = {name: r['metrics'] for name, r in cached_results.items()}
                
                return cached_results
        
        # Train all models if cache not available or force_retrain
        results = {}
        print(f"Training {len(self.models)} models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            }
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                metrics['cv_mean'] = float(cv_scores.mean())
                metrics['cv_std'] = float(cv_scores.std())
            except:
                metrics['cv_mean'] = metrics['accuracy']
                metrics['cv_std'] = 0.0
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None and hasattr(y_pred_proba, 'tolist') else None
            }
            
            self.trained_models[name] = model
            self.model_metrics[name] = metrics
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
        self.best_model = results[best_model_name]['model']
        
        # Mark as not from cache
        for name in results:
            results[name]['from_cache'] = False
        
        return results
    
    def save_comparison_cache(self, cache_key, results, vectorizer):
        """Save comparison results to cache"""
        self._save_cached_models(cache_key, results, vectorizer)
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        base_model = self.models[model_name]
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, 
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'best_model': grid_search.best_estimator_
        }
    
    def create_ensemble(self, selected_models, X_train, y_train, voting='soft'):
        """Create an ensemble of selected models"""
        estimators = [(name, self.trained_models[name]) for name in selected_models if name in self.trained_models]
        
        if not estimators:
            raise ValueError("No trained models available for ensemble")
        
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def save_model(self, model, model_name, vectorizer, accuracy, metrics):
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_file = f'models/{model_name}_{timestamp}.pkl'
        vectorizer_file = f'models/{model_name}_vectorizer_{timestamp}.pkl'
        info_file = f'models/{model_name}_info_{timestamp}.json'
        
        os.makedirs('models', exist_ok=True)
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        model_info = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'metrics': metrics,
            'trained_date': datetime.now().isoformat(),
            'model_file': model_file,
            'vectorizer_file': vectorizer_file
        }
        
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Also save as latest
        with open(f'models/{model_name}_latest.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(f'models/{model_name}_vectorizer_latest.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return model_info
    
    def load_model(self, model_name, use_latest=True, timestamp=None):
        """Load a saved model"""
        if use_latest:
            model_file = f'models/{model_name}_latest.pkl'
            vectorizer_file = f'models/{model_name}_vectorizer_latest.pkl'
        else:
            model_file = f'models/{model_name}_{timestamp}.pkl'
            vectorizer_file = f'models/{model_name}_vectorizer_{timestamp}.pkl'
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer

# Parameter grids for hyperparameter tuning
PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'gradient_boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'neural_network': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}

