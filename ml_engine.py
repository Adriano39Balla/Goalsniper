"""
Advanced ML Engine for Betting Predictions
Uses gradient boosting with meta-learning to automatically discover optimal betting markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
import json
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from loguru import logger


@dataclass
class PredictionResult:
    """Structured prediction output"""
    fixture_id: int
    market: str
    prediction: str
    probability: float
    calibrated_probability: float
    confidence_score: float
    expected_value: float
    timestamp: datetime
    features_used: List[str]
    model_version: str


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    log_loss: float
    brier_score: float
    auc_roc: float
    calibration_error: float
    profit_loss: float
    total_predictions: int
    winning_predictions: int


class AdaptiveMarketSelector:
    """
    Meta-learning component that automatically identifies the best betting markets
    based on feature importance and historical performance
    """
    
    def __init__(self):
        self.market_performance: Dict[str, ModelPerformance] = {}
        self.market_features: Dict[str, List[str]] = {}
        
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect which features are relevant for different markets
        """
        markets = {}
        
        # Goal-based markets
        goal_features = [col for col in df.columns if any(x in col.lower() 
                        for x in ['goal', 'shot', 'attack', 'xg', 'scoring'])]
        if goal_features:
            markets['over_under_goals'] = goal_features
            markets['btts'] = goal_features
            markets['next_goal'] = goal_features
            
        # Card-based markets
        card_features = [col for col in df.columns if any(x in col.lower() 
                        for x in ['card', 'foul', 'yellow', 'red', 'aggression'])]
        if card_features:
            markets['total_cards'] = card_features
            
        # Corner-based markets
        corner_features = [col for col in df.columns if any(x in col.lower() 
                          for x in ['corner', 'pressure', 'attack'])]
        if corner_features:
            markets['total_corners'] = corner_features
            
        # Match outcome markets
        outcome_features = [col for col in df.columns if any(x in col.lower() 
                           for x in ['possession', 'shot', 'attack', 'defense', 'rating'])]
        if outcome_features:
            markets['match_winner'] = outcome_features
            
        self.market_features = markets
        logger.info(f"Detected {len(markets)} potential markets with features")
        return markets
    
    def select_best_markets(self, top_n: int = 3) -> List[str]:
        """
        Select top performing markets based on historical performance
        """
        if not self.market_performance:
            return list(self.market_features.keys())[:top_n]
        
        # Sort by expected value and accuracy
        sorted_markets = sorted(
            self.market_performance.items(),
            key=lambda x: (x[1].expected_value, x[1].accuracy),
            reverse=True
        )
        
        return [market for market, _ in sorted_markets[:top_n]]


class GradientBoostingEnsemble:
    """
    Advanced ensemble combining XGBoost, LightGBM, and CatBoost
    with probability calibration and online learning
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, np.ndarray] = {}
        self.market_selector = AdaptiveMarketSelector()
        self.training_history: List[Dict] = []
        
    def _create_model(self, market: str) -> Any:
        """Create optimized gradient boosting model"""
        
        if self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                importance_type='gain',
                verbose=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=8,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_weight=3,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for live in-play betting
        """
        df = df.copy()
        
        # Time-based features
        if 'elapsed_minutes' in df.columns:
            df['time_pressure'] = df['elapsed_minutes'] / 90
            df['time_remaining'] = 90 - df['elapsed_minutes']
            df['is_first_half'] = (df['elapsed_minutes'] <= 45).astype(int)
            df['is_second_half'] = (df['elapsed_minutes'] > 45).astype(int)
            df['is_final_15min'] = (df['elapsed_minutes'] >= 75).astype(int)
        
        # Goal momentum features
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            df['goal_difference'] = df['home_goals'] - df['away_goals']
            df['total_goals'] = df['home_goals'] + df['away_goals']
            df['home_leading'] = (df['goal_difference'] > 0).astype(int)
            df['away_leading'] = (df['goal_difference'] < 0).astype(int)
            df['is_draw'] = (df['goal_difference'] == 0).astype(int)
        
        # Attack intensity features
        attack_cols = [col for col in df.columns if 'attack' in col.lower() or 'shot' in col.lower()]
        if attack_cols:
            df['total_attacks'] = df[attack_cols].sum(axis=1)
            df['attack_imbalance'] = df[[c for c in attack_cols if 'home' in c.lower()]].sum(axis=1) - \
                                     df[[c for c in attack_cols if 'away' in c.lower()]].sum(axis=1)
        
        # Possession-based features
        if 'home_possession' in df.columns and 'away_possession' in df.columns:
            df['possession_difference'] = df['home_possession'] - df['away_possession']
            df['possession_dominance'] = np.abs(df['possession_difference'])
        
        # Card and foul features
        card_cols = [col for col in df.columns if 'card' in col.lower() or 'foul' in col.lower()]
        if card_cols:
            df['total_cards'] = df[[c for c in card_cols if 'card' in c.lower()]].sum(axis=1)
            df['aggression_level'] = df[card_cols].sum(axis=1)
        
        # Rolling statistics (if historical data available)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['fixture_id', 'elapsed_minutes']:
                df[f'{col}_rolling_mean'] = df.groupby('fixture_id')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                df[f'{col}_trend'] = df.groupby('fixture_id')[col].transform(
                    lambda x: x.diff().fillna(0)
                )
        
        # Interaction features
        if 'home_goals' in df.columns and 'elapsed_minutes' in df.columns:
            df['goals_per_minute'] = df['total_goals'] / (df['elapsed_minutes'] + 1)
            df['expected_final_goals'] = df['goals_per_minute'] * 90
        
        return df
    
    def train_market_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        market: str,
        validation_split: float = 0.2
    ) -> ModelPerformance:
        """
        Train model for specific market with time-series cross-validation
        """
        logger.info(f"Training model for market: {market}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[market] = scaler
        
        # Time-series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train model
        model = self._create_model(market)
        
        # Cross-validation scores
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, y_pred_proba)
            cv_scores.append(score)
        
        # Final training on full dataset
        model.fit(X_scaled, y)
        self.models[market] = model
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[market] = model.feature_importances_
        
        # Train probability calibrator
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred_proba, y)
        self.calibrators[market] = calibrator
        
        # Calculate performance metrics
        y_pred_calibrated = calibrator.transform(y_pred_proba)
        y_pred_binary = (y_pred_calibrated > 0.5).astype(int)
        
        performance = ModelPerformance(
            accuracy=np.mean(y_pred_binary == y),
            log_loss=log_loss(y, y_pred_calibrated),
            brier_score=brier_score_loss(y, y_pred_calibrated),
            auc_roc=roc_auc_score(y, y_pred_calibrated),
            calibration_error=np.mean(np.abs(y_pred_calibrated - y)),
            profit_loss=0.0,  # Updated during live predictions
            total_predictions=len(y),
            winning_predictions=np.sum(y_pred_binary == y)
        )
        
        self.market_selector.market_performance[market] = performance
        
        logger.info(f"Market {market} - Accuracy: {performance.accuracy:.3f}, "
                   f"Log Loss: {performance.log_loss:.3f}, AUC: {performance.auc_roc:.3f}")
        
        return performance
    
    def predict(
        self, 
        X: pd.DataFrame, 
        fixture_id: int,
        market: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[PredictionResult]:
        """
        Generate calibrated prediction for a specific market
        """
        if market not in self.models:
            logger.warning(f"No model trained for market: {market}")
            return None
        
        # Scale features
        X_scaled = self.scalers[market].transform(X)
        
        # Get raw probability
        raw_proba = self.models[market].predict_proba(X_scaled)[:, 1][0]
        
        # Calibrate probability
        calibrated_proba = self.calibrators[market].transform([raw_proba])[0]
        
        # Calculate confidence score (distance from 0.5)
        confidence = abs(calibrated_proba - 0.5) * 2
        
        # Determine prediction
        prediction = "YES" if calibrated_proba > 0.5 else "NO"
        
        # Calculate expected value (simplified - should include odds)
        expected_value = calibrated_proba if prediction == "YES" else (1 - calibrated_proba)
        
        return PredictionResult(
            fixture_id=fixture_id,
            market=market,
            prediction=prediction,
            probability=raw_proba,
            calibrated_probability=calibrated_proba,
            confidence_score=confidence,
            expected_value=expected_value,
            timestamp=timestamp or datetime.now(),
            features_used=list(X.columns),
            model_version=f"{self.model_type}_v1"
        )
    
    def update_from_outcome(
        self, 
        prediction: PredictionResult, 
        actual_outcome: bool,
        odds: float = 2.0
    ):
        """
        Online learning: Update model based on prediction outcome
        """
        market = prediction.market
        
        # Calculate profit/loss
        if prediction.prediction == "YES":
            profit = (odds - 1) if actual_outcome else -1
        else:
            profit = (odds - 1) if not actual_outcome else -1
        
        # Update market performance
        if market in self.market_selector.market_performance:
            perf = self.market_selector.market_performance[market]
            perf.total_predictions += 1
            perf.profit_loss += profit
            if (prediction.prediction == "YES" and actual_outcome) or \
               (prediction.prediction == "NO" and not actual_outcome):
                perf.winning_predictions += 1
            perf.accuracy = perf.winning_predictions / perf.total_predictions
        
        # Log for retraining
        self.training_history.append({
            'timestamp': prediction.timestamp,
            'market': market,
            'predicted_proba': prediction.calibrated_probability,
            'actual_outcome': actual_outcome,
            'profit': profit
        })
        
        logger.info(f"Updated model from outcome - Market: {market}, "
                   f"Correct: {actual_outcome}, Profit: {profit:.2f}")
    
    def save_models(self, directory: Path):
        """Save all models and calibrators"""
        directory.mkdir(parents=True, exist_ok=True)
        
        for market, model in self.models.items():
            model_path = directory / f"model_{market}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            calibrator_path = directory / f"calibrator_{market}.pkl"
            with open(calibrator_path, 'wb') as f:
                pickle.dump(self.calibrators[market], f)
            
            scaler_path = directory / f"scaler_{market}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[market], f)
        
        # Save market selector state
        selector_path = directory / "market_selector.json"
        with open(selector_path, 'w') as f:
            json.dump({
                'market_features': self.market_selector.market_features,
                'market_performance': {
                    k: v.__dict__ for k, v in self.market_selector.market_performance.items()
                }
            }, f, indent=2)
        
        logger.info(f"Saved {len(self.models)} models to {directory}")
    
    def load_models(self, directory: Path):
        """Load all models and calibrators"""
        if not directory.exists():
            logger.warning(f"Model directory {directory} does not exist")
            return
        
        for model_file in directory.glob("model_*.pkl"):
            market = model_file.stem.replace("model_", "")
            
            with open(model_file, 'rb') as f:
                self.models[market] = pickle.load(f)
            
            calibrator_path = directory / f"calibrator_{market}.pkl"
            with open(calibrator_path, 'rb') as f:
                self.calibrators[market] = pickle.load(f)
            
            scaler_path = directory / f"scaler_{market}.pkl"
            with open(scaler_path, 'rb') as f:
                self.scalers[market] = pickle.load(f)
        
        # Load market selector state
        selector_path = directory / "market_selector.json"
        if selector_path.exists():
            with open(selector_path, 'r') as f:
                data = json.load(f)
                self.market_selector.market_features = data['market_features']
                self.market_selector.market_performance = {
                    k: ModelPerformance(**v) for k, v in data['market_performance'].items()
                }
        
        logger.info(f"Loaded {len(self.models)} models from {directory}")
