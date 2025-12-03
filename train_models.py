#!/usr/bin/env python3
"""
Model training module with 80/20 train-test split
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from utils.logger import logger
from utils.database import DatabaseManager
from models.random_forest import BettingPredictor, EnhancedRandomForest
from utils.feature_engineering import FeatureEngineer

class ModelTrainer:
    """Model trainer with comprehensive training pipeline"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.predictor = BettingPredictor()
        self.feature_engineer = FeatureEngineer()
        self.scalers = {}
        self.label_encoders = {}
        
    def prepare_training_data(self, limit: int = 20000) -> Tuple[Dict, Dict, Dict, Dict]:
        """Prepare training and testing data with 80/20 split"""
        
        logger.info("Preparing training data...")
        
        # Get historical data
        training_data = self.db.get_training_data(limit=limit)
        
        if len(training_data) < 1000:
            logger.warning(f"Insufficient training data: {len(training_data)} records")
            return {}, {}, {}, {}
        
        # Separate data by prediction type
        data_by_type = {'1X2': [], 'over_under': [], 'btts': []}
        
        for record in training_data:
            pred_type = record['prediction_type']
            if pred_type in data_by_type:
                data_by_type[pred_type].append(record)
        
        # Prepare features and labels for each type
        X_train_all, X_test_all, y_train_all, y_test_all = {}, {}, {}, {}
        
        for pred_type, records in data_by_type.items():
            if len(records) < 100:
                logger.warning(f"Insufficient {pred_type} records: {len(records)}")
                continue
            
            # Extract features and labels
            X, y = self.extract_features_labels(records, pred_type)
            
            if len(X) == 0:
                continue
            
            # Split data 80/20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[pred_type] = scaler
            
            # Store prepared data
            X_train_all[pred_type] = X_train_scaled
            X_test_all[pred_type] = X_test_scaled
            y_train_all[pred_type] = y_train
            y_test_all[pred_type] = y_test
            
            logger.info(f"{pred_type}: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train_all, X_test_all, y_train_all, y_test_all
    
    def extract_features_labels(self, records: List[Dict], pred_type: str):
        """Extract features and labels from records"""
        
        features = []
        labels = []
        
        for record in records:
            # Extract features from stored JSON
            raw_features = record.get('features', {})
            
            if not raw_features or pred_type not in raw_features:
                continue
            
            # Get features for specific prediction type
            feature_vector = raw_features.get(pred_type)
            
            if feature_vector is None:
                continue
            
            # Convert to numpy array
            try:
                if isinstance(feature_vector, list):
                    feature_array = np.array(feature_vector)
                else:
                    feature_array = np.array([feature_vector])
                
                features.append(feature_array)
                
                # Get label
                actual_result = record.get('actual_result')
                if actual_result:
                    labels.append(actual_result)
                    
            except Exception as e:
                logger.debug(f"Error processing features: {e}")
                continue
        
        if not features or not labels:
            return np.array([]), np.array([])
        
        # Encode labels
        if pred_type not in self.label_encoders:
            self.label_encoders[pred_type] = LabelEncoder()
        
        le = self.label_encoders[pred_type]
        labels_encoded = le.fit_transform(labels)
        
        # Convert to numpy arrays
        X = np.vstack(features)
        y = np.array(labels_encoded)
        
        return X, y
    
    def train_all_models(self, optimize: bool = True):
        """Train all prediction models"""
        
        logger.info("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        if not X_train:
            logger.error("No training data available")
            return
        
        # Train models
        results = self.predictor.train_all(X_train, y_train, X_test, y_test)
        
        # Save scalers and encoders
        self.save_preprocessors()
        
        # Log training results
        self.log_training_results(results)
        
        # Save performance metrics to database
        self.save_performance_metrics(results)
        
        logger.info("Model training completed")
        return results
    
    def save_preprocessors(self):
        """Save scalers and label encoders"""
        
        preprocessors_dir = Path('data/models/preprocessors')
        preprocessors_dir.mkdir(exist_ok=True)
        
        # Save scalers
        for pred_type, scaler in self.scalers.items():
            joblib.dump(scaler, preprocessors_dir / f"{pred_type}_scaler.joblib")
        
        # Save label encoders
        for pred_type, encoder in self.label_encoders.items():
            joblib.dump(encoder, preprocessors_dir / f"{pred_type}_encoder.joblib")
        
        logger.info("Preprocessors saved")
    
    def log_training_results(self, results: Dict[str, Dict]):
        """Log detailed training results"""
        
        logger.info("=== Training Results ===")
        
        for model_type, metrics in results.items():
            logger.info(f"\n{model_type} Model:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Cross-validation scores
            if model_type in self.predictor.models:
                model = self.predictor.models[model_type]
                if model.model and hasattr(model, 'classes_') and len(model.classes_) > 0:
                    # Perform cross-validation
                    X = np.vstack([self.X_train_all.get(model_type, []), 
                                  self.X_test_all.get(model_type, [])])
                    y = np.concatenate([self.y_train_all.get(model_type, []), 
                                       self.y_test_all.get(model_type, [])])
                    
                    if len(X) > 0 and len(y) > 0:
                        cv_scores = cross_val_score(
                            model.model, X, y, cv=5, scoring='accuracy'
                        )
                        logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    def save_performance_metrics(self, results: Dict[str, Dict]):
        """Save performance metrics to database"""
        
        for model_type, metrics in results.items():
            query = """
            INSERT INTO model_performance (
                model_type, version, accuracy, precision,
                recall, f1_score, roc_auc, parameters,
                feature_importance, training_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                model_type,
                os.getenv('MODEL_VERSION', 'v1.0'),
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('roc_auc', 0),
                Json(self.predictor.models[model_type].best_params 
                     if hasattr(self.predictor.models[model_type], 'best_params') 
                     else {}),
                Json(self.predictor.models[model_type].feature_importance 
                     if hasattr(self.predictor.models[model_type], 'feature_importance') 
                     else {}),
                datetime.utcnow()
            )
            
            self.db.execute_query(query, params)
        
        logger.info("Performance metrics saved to database")
    
    def evaluate_models(self):
        """Evaluate models on test set"""
        
        logger.info("Evaluating models...")
        
        # Load test data
        _, X_test, _, y_test = self.prepare_training_data(limit=5000)
        
        if not X_test:
            logger.warning("No test data available")
            return
        
        evaluations = {}
        
        for model_type, model in self.predictor.models.items():
            if model.model is None:
                continue
            
            X_test_data = X_test.get(model_type)
            y_test_data = y_test.get(model_type)
            
            if X_test_data is None or y_test_data is None:
                continue
            
            # Make predictions
            y_pred = model.predict(X_test_data)
            y_pred_proba = model.predict_proba(X_test_data)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test_data, y_pred),
                'precision': precision_score(y_test_data, y_pred, average='weighted'),
                'recall': recall_score(y_test_data, y_pred, average='weighted'),
                'f1': f1_score(y_test_data, y_pred, average='weighted')
            }
            
            # Add ROC AUC for binary classification
            if len(model.classes_) == 2:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc'] = roc_auc_score(y_test_data, y_pred_proba[:, 1])
            
            evaluations[model_type] = metrics
            
            # Log detailed classification report
            logger.info(f"\n{classification_report(y_test_data, y_pred)}")
            
            # Log confusion matrix
            cm = confusion_matrix(y_test_data, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
        
        return evaluations
    
    def optimize_thresholds(self):
        """Optimize prediction thresholds for maximum profit"""
        
        logger.info("Optimizing prediction thresholds...")
        
        # This would analyze historical predictions to find optimal thresholds
        # Implementation depends on your specific profit optimization goals
        
        # For now, use ROC curve to find optimal thresholds
        optimal_thresholds = {}
        
        for model_type, model in self.predictor.models.items():
            if model.model is None:
                continue
            
            # Load validation data
            _, X_val, _, y_val = self.prepare_training_data(limit=3000)
            
            X_val_data = X_val.get(model_type)
            y_val_data = y_val.get(model_type)
            
            if X_val_data is None or y_val_data is None:
                continue
            
            # Get probabilities
            y_proba = model.predict_proba(X_val_data)
            
            if len(model.classes_) == 2:
                # Find optimal threshold using Youden's J statistic
                from sklearn.metrics import roc_curve
                
                fpr, tpr, thresholds = roc_curve(y_val_data, y_proba[:, 1])
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                optimal_thresholds[model_type] = optimal_threshold
                
                logger.info(f"{model_type} optimal threshold: {optimal_threshold:.3f}")
        
        # Save optimal thresholds
        thresholds_path = Path('data/models/optimal_thresholds.json')
        import json
        with open(thresholds_path, 'w') as f:
            json.dump(optimal_thresholds, f)
        
        logger.info(f"Optimal thresholds saved to {thresholds_path}")
        return optimal_thresholds

def main():
    """Main training function"""
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize
    db = DatabaseManager()
    trainer = ModelTrainer(db)
    
    # Create tables if needed
    db.create_tables()
    
    # Train models
    trainer.train_all_models(optimize=True)
    
    # Evaluate
    trainer.evaluate_models()
    
    # Optimize thresholds
    trainer.optimize_thresholds()
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()
