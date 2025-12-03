import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional, Any
import joblib
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger

class EnhancedRandomForest:
    """Enhanced Random Forest for betting predictions with calibration"""
    
    def __init__(self, model_type: str = '1X2'):
        self.model_type = model_type  # '1X2', 'OU', 'BTTS'
        self.model = None
        self.calibrator = None
        self.feature_importance = None
        self.classes_ = None
        self.best_params = None
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', 
                                                         ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', 
                                                         ['balanced', 'balanced_subsample']),
                'random_state': 42
            }
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                
                # Use probability for AUC
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Optimized parameters for {self.model_type}: {self.best_params}")
        
        return self.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: Optional[np.ndarray] = None,
              y_test: Optional[np.ndarray] = None,
              optimize: bool = True) -> Dict[str, float]:
        """Train the Random Forest model with calibration"""
        
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train), 
                                            y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Optimize or use default
        if optimize:
            params = self.optimize_hyperparameters(X_train, y_train)
        else:
            params = {
                'n_estimators': 300,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Create and train base model
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Calibrate probabilities
        self.calibrator = CalibratedClassifierCV(self.model, 
                                                method='isotonic', 
                                                cv='prefit')
        self.calibrator.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            range(len(self.model.feature_importances_)),
            self.model.feature_importances_
        ))
        
        # Evaluate
        metrics = {}
        if X_test is not None and y_test is not None:
            metrics = self.evaluate(X_test, y_test)
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities"""
        if self.calibrator:
            return self.calibrator.predict_proba(X)
        elif self.model:
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model not trained")
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict with threshold"""
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:  # Binary classification
            return (proba[:, 1] >= threshold).astype(int)
        else:  # Multiclass
            return np.argmax(proba, axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # ROC AUC for binary classification
        if len(self.classes_) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Log metrics
        logger.info(f"Model {self.model_type} evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'classes': self.classes_,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'model_type': self.model_type
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.calibrator = model_data['calibrator']
        self.classes_ = model_data['classes']
        self.feature_importance = model_data['feature_importance']
        self.best_params = model_data['best_params']
        self.model_type = model_data['model_type']
        logger.info(f"Model loaded from {path}")

class BettingPredictor:
    """Main predictor combining multiple models"""
    
    def __init__(self):
        self.models = {
            '1X2': EnhancedRandomForest('1X2'),
            'over_under': EnhancedRandomForest('OU'),
            'btts': EnhancedRandomForest('BTTS')
        }
        self.feature_names = None
        
    def train_all(self, X_train: Dict[str, np.ndarray], 
                  y_train: Dict[str, np.ndarray],
                  X_test: Optional[Dict[str, np.ndarray]] = None,
                  y_test: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict]:
        """Train all prediction models"""
        
        results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model...")
            
            X_train_model = X_train.get(model_type)
            y_train_model = y_train.get(model_type)
            
            if X_train_model is None or y_train_model is None:
                logger.warning(f"No data for {model_type}, skipping")
                continue
            
            # Get test data if provided
            X_test_model = X_test.get(model_type) if X_test else None
            y_test_model = y_test.get(model_type) if y_test else None
            
            # Train model
            metrics = model.train(X_train_model, y_train_model,
                                 X_test_model, y_test_model)
            
            results[model_type] = metrics
            
            # Save model
            model.save(f"data/models/{model_type}_model.joblib")
        
        return results
    
    def predict_match(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Predict all outcomes for a match"""
        
        predictions = {}
        
        for model_type, model in self.models.items():
            if model.model is None:
                logger.warning(f"Model {model_type} not trained")
                continue
            
            model_features = features.get(model_type)
            if model_features is None:
                continue
            
            # Get probabilities
            proba = model.predict_proba(model_features.reshape(1, -1))[0]
            
            if model_type == '1X2':
                # Home, Draw, Away probabilities
                if len(proba) == 3:
                    predictions['1X2'] = {
                        'home_win': float(proba[0]),
                        'draw': float(proba[1]),
                        'away_win': float(proba[2]),
                        'prediction': model.classes_[np.argmax(proba)]
                    }
            elif model_type == 'over_under':
                # Over/Under 2.5 goals
                predictions['over_under'] = {
                    'over': float(proba[0]),
                    'under': float(proba[1]),
                    'prediction': 'over' if proba[0] > proba[1] else 'under'
                }
            elif model_type == 'btts':
                # Both teams to score
                predictions['btts'] = {
                    'yes': float(proba[0]),
                    'no': float(proba[1]),
                    'prediction': 'yes' if proba[0] > proba[1] else 'no'
                }
        
        return predictions
