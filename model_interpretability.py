"""
Model Interpretability Module
Provides SHAP values, feature importance, and LIME explanations
"""

import numpy as np
from sklearn.inspection import permutation_importance
import pandas as pd

class ModelInterpreter:
    """Provides model interpretability and explainability"""
    
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
    
    def get_feature_importance(self, X_train, y_train, n_features=20):
        """Get feature importance using permutation importance or coefficients"""
        try:
            # For tree-based models, use built-in feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.vectorizer.get_feature_names_out()
                importances = self.model.feature_importances_
                
                indices = np.argsort(importances)[::-1][:n_features]
                
                return [
                    {
                        'feature': feature_names[i],
                        'importance': float(importances[i])
                    }
                    for i in indices
                ]
            
            # For Logistic Regression and linear models, use coefficient magnitudes (faster)
            if hasattr(self.model, 'coef_'):
                feature_names = self.vectorizer.get_feature_names_out()
                # Get average absolute coefficient across all classes
                coefs = np.abs(self.model.coef_).mean(axis=0)
                
                indices = np.argsort(coefs)[::-1][:n_features]
                
                return [
                    {
                        'feature': feature_names[i],
                        'importance': float(coefs[i])
                    }
                    for i in indices
                ]
            
            # For other models, use permutation importance (slower, so use sample)
            # Sample data if too large to speed up calculation
            if X_train.shape[0] > 1000:
                sample_indices = np.random.choice(X_train.shape[0], 1000, replace=False)
                X_sample = X_train[sample_indices]
                y_sample = y_train.iloc[sample_indices] if hasattr(y_train, 'iloc') else y_train[sample_indices]
            else:
                X_sample = X_train
                y_sample = y_train
            
            perm_importance = permutation_importance(
                self.model, X_sample, y_sample, 
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            feature_names = self.vectorizer.get_feature_names_out()
            importances = perm_importance.importances_mean
            
            indices = np.argsort(importances)[::-1][:n_features]
            
            return [
                {
                    'feature': feature_names[i],
                    'importance': float(importances[i]),
                    'std': float(perm_importance.importances_std[i])
                }
                for i in indices
            ]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error calculating feature importance: {e}")
            return []
    
    def explain_prediction(self, text, top_features=10):
        """Explain a single prediction by showing important features"""
        try:
            # Transform text
            text_tfidf = self.vectorizer.transform([text])
            
            # Get prediction
            prediction = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get coefficients/weights for the predicted class
            if hasattr(self.model, 'coef_'):
                class_idx = list(self.model.classes_).index(prediction)
                coefs = self.model.coef_[class_idx]
                
                # Get feature indices present in the text
                text_features = text_tfidf.toarray()[0]
                non_zero_indices = np.nonzero(text_features)[0]
                
                # Get top contributing features
                feature_scores = []
                for idx in non_zero_indices:
                    score = coefs[idx] * text_features[idx]
                    feature_scores.append({
                        'feature': feature_names[idx],
                        'coefficient': float(coefs[idx]),
                        'tfidf_value': float(text_features[idx]),
                        'contribution': float(score)
                    })
                
                feature_scores.sort(key=lambda x: abs(x['contribution']), reverse=True)
                
                return {
                    'prediction': prediction,
                    'probabilities': {cls: float(prob) for cls, prob in zip(self.model.classes_, probabilities)},
                    'top_features': feature_scores[:top_features],
                    'positive_contributors': [f for f in feature_scores if f['contribution'] > 0][:5],
                    'negative_contributors': [f for f in feature_scores if f['contribution'] < 0][:5]
                }
            else:
                # For tree-based models, use a different approach
                return {
                    'prediction': prediction,
                    'probabilities': {cls: float(prob) for cls, prob in zip(self.model.classes_, probabilities)},
                    'message': 'Detailed explanation not available for this model type'
                }
        except Exception as e:
            print(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def get_shap_values(self, X_sample, n_samples=100):
        """Calculate SHAP values (simplified version)"""
        try:
            import shap
            from shap import Explainer
            
            # Sample data if too large
            if X_sample.shape[0] > n_samples:
                indices = np.random.choice(X_sample.shape[0], n_samples, replace=False)
                X_sample = X_sample[indices]
            
            explainer = Explainer(self.model, X_sample)
            shap_values = explainer(X_sample)
            
            return {
                'shap_values': shap_values.values.tolist() if hasattr(shap_values.values, 'tolist') else shap_values.values,
                'base_value': float(shap_values.base_values) if hasattr(shap_values, 'base_values') else None
            }
        except ImportError:
            return {'error': 'SHAP library not installed. Install with: pip install shap'}
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_prediction_confidence(self, texts):
        """Analyze prediction confidence across a dataset"""
        try:
            predictions = self.model.predict(self.vectorizer.transform(texts))
            probabilities = self.model.predict_proba(self.vectorizer.transform(texts))
            
            confidences = []
            for prob in probabilities:
                max_prob = np.max(prob)
                confidences.append(float(max_prob))
            
            return {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'low_confidence_count': sum(1 for c in confidences if c < 0.7),
                'high_confidence_count': sum(1 for c in confidences if c > 0.9)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_class_wise_feature_importance(self, X_train, y_train):
        """Get feature importance for each class"""
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            classes = self.model.classes_
            
            class_importance = {}
            
            if hasattr(self.model, 'coef_'):
                for i, class_name in enumerate(classes):
                    coefs = self.model.coef_[i]
                    indices = np.argsort(np.abs(coefs))[::-1][:10]
                    
                    class_importance[class_name] = [
                        {
                            'feature': feature_names[idx],
                            'coefficient': float(coefs[idx])
                        }
                        for idx in indices
                    ]
            
            return class_importance
        except Exception as e:
            return {'error': str(e)}

