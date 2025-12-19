"""
Model Training and Testing Module
Handles data preparation, model training, validation, and performance evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import asyncio

from loguru import logger
from ml_engine import GradientBoostingEnsemble, ModelPerformance
from data_pipeline import FootballDataPipeline
from database import DatabaseManager
from config import settings


class ModelTrainer:
    """
    Comprehensive model training system with automated data preparation,
    feature engineering, and performance validation
    """
    
    def __init__(self):
        self.ml_engine = GradientBoostingEnsemble(model_type=settings.MODEL_TYPE)
        self.data_pipeline = FootballDataPipeline()
        self.db_manager = DatabaseManager()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    async def backfill_historical_data(self, days: int = 90) -> pd.DataFrame:
        """
        Backfill historical match data for training
        """
        logger.info(f"Backfilling {days} days of historical data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_matches = []
        
        # Fetch data day by day to avoid API limits
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"Fetching matches for {date_str}")
            
            try:
                matches = await self.data_pipeline.fetch_live_matches(date=date_str)
                if matches:
                    all_matches.extend(matches)
                    logger.info(f"Found {len(matches)} matches on {date_str}")
                
                # Respect API rate limits
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching data for {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_matches:
            logger.warning("No historical data retrieved")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_matches)
        logger.info(f"Backfilled {len(df)} match records")
        
        # Save to database
        await self.db_manager.save_historical_matches(df)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare training datasets for different betting markets
        """
        logger.info("Preparing training data for multiple markets...")
        
        # Engineer features
        df_engineered = self.ml_engine.engineer_features(df)
        
        # Analyze and detect markets
        self.ml_engine.market_selector.analyze_features(df_engineered)
        
        training_datasets = {}
        
        # Prepare data for each market
        markets_config = {
            'over_under_goals': {
                'target_col': 'total_goals',
                'threshold': 2.5,
                'condition': lambda x: x > 2.5
            },
            'btts': {
                'target_col': ['home_goals', 'away_goals'],
                'condition': lambda df: (df['home_goals'] > 0) & (df['away_goals'] > 0)
            },
            'next_goal': {
                'target_col': 'next_goal_team',
                'condition': lambda x: x == 'home'
            },
            'total_cards': {
                'target_col': 'total_cards',
                'threshold': 4.5,
                'condition': lambda x: x > 4.5
            },
            'total_corners': {
                'target_col': 'total_corners',
                'threshold': 9.5,
                'condition': lambda x: x > 9.5
            }
        }
        
        for market, config in markets_config.items():
            try:
                # Get relevant features for this market
                if market in self.ml_engine.market_selector.market_features:
                    feature_cols = self.ml_engine.market_selector.market_features[market]
                else:
                    # Use all numeric features if market not detected
                    feature_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
                
                # Remove target columns from features
                target_col = config['target_col']
                if isinstance(target_col, str):
                    feature_cols = [c for c in feature_cols if c != target_col]
                else:
                    feature_cols = [c for c in feature_cols if c not in target_col]
                
                # Create target variable
                if callable(config['condition']):
                    if isinstance(target_col, list):
                        y = config['condition'](df_engineered).astype(int)
                    else:
                        y = config['condition'](df_engineered[target_col]).astype(int)
                else:
                    y = (df_engineered[target_col] > config['threshold']).astype(int)
                
                # Filter valid features
                X = df_engineered[feature_cols].copy()
                
                # Remove rows with missing values
                valid_idx = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_idx]
                y = y[valid_idx]
                
                if len(X) > 100:  # Minimum samples required
                    training_datasets[market] = (X, y)
                    logger.info(f"Prepared {len(X)} samples for market: {market}")
                else:
                    logger.warning(f"Insufficient data for market {market}: {len(X)} samples")
                    
            except Exception as e:
                logger.error(f"Error preparing data for market {market}: {e}")
        
        return training_datasets
    
    async def train_all_markets(self, df: pd.DataFrame) -> Dict[str, ModelPerformance]:
        """
        Train models for all detected markets
        """
        logger.info("Starting comprehensive model training...")
        
        # Prepare training data
        training_datasets = self.prepare_training_data(df)
        
        if not training_datasets:
            logger.error("No training datasets prepared")
            return {}
        
        # Train models for each market
        performance_results = {}
        
        for market, (X, y) in training_datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training market: {market}")
            logger.info(f"{'='*60}")
            
            try:
                performance = self.ml_engine.train_market_model(X, y, market)
                performance_results[market] = performance
                
                # Log detailed performance
                logger.info(f"\nPerformance Metrics for {market}:")
                logger.info(f"  Accuracy: {performance.accuracy:.4f}")
                logger.info(f"  Log Loss: {performance.log_loss:.4f}")
                logger.info(f"  Brier Score: {performance.brier_score:.4f}")
                logger.info(f"  AUC-ROC: {performance.auc_roc:.4f}")
                logger.info(f"  Calibration Error: {performance.calibration_error:.4f}")
                logger.info(f"  Total Predictions: {performance.total_predictions}")
                logger.info(f"  Winning Predictions: {performance.winning_predictions}")
                
            except Exception as e:
                logger.error(f"Error training market {market}: {e}")
        
        # Save trained models
        self.ml_engine.save_models(self.models_dir)
        
        # Save performance metrics to database
        await self.db_manager.save_model_performance(performance_results)
        
        # Select best markets
        best_markets = self.ml_engine.market_selector.select_best_markets(top_n=3)
        logger.info(f"\nBest performing markets: {best_markets}")
        
        return performance_results
    
    async def validate_models(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Validate trained models on unseen test data
        """
        logger.info("Validating models on test data...")
        
        validation_results = {}
        
        # Prepare test data
        test_datasets = self.prepare_training_data(test_df)
        
        for market, (X_test, y_test) in test_datasets.items():
            if market not in self.ml_engine.models:
                logger.warning(f"No model found for market {market}")
                continue
            
            try:
                # Generate predictions
                predictions = []
                for idx in range(len(X_test)):
                    X_sample = X_test.iloc[[idx]]
                    pred = self.ml_engine.predict(
                        X_sample, 
                        fixture_id=idx,
                        market=market
                    )
                    if pred:
                        predictions.append(pred.calibrated_probability)
                
                if predictions:
                    predictions = np.array(predictions)
                    
                    # Calculate validation metrics
                    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
                    
                    val_metrics = {
                        'log_loss': log_loss(y_test, predictions),
                        'brier_score': brier_score_loss(y_test, predictions),
                        'auc_roc': roc_auc_score(y_test, predictions),
                        'accuracy': np.mean((predictions > 0.5) == y_test),
                        'samples': len(predictions)
                    }
                    
                    validation_results[market] = val_metrics
                    
                    logger.info(f"\nValidation Results for {market}:")
                    logger.info(f"  Test Accuracy: {val_metrics['accuracy']:.4f}")
                    logger.info(f"  Test Log Loss: {val_metrics['log_loss']:.4f}")
                    logger.info(f"  Test AUC-ROC: {val_metrics['auc_roc']:.4f}")
                    
            except Exception as e:
                logger.error(f"Error validating market {market}: {e}")
        
        return validation_results
    
    async def auto_tune_models(self) -> Dict[str, ModelPerformance]:
        """
        Automatically tune hyperparameters based on recent performance
        """
        logger.info("Starting automatic hyperparameter tuning...")
        
        # Get recent performance data
        recent_predictions = await self.db_manager.get_recent_predictions(days=7)
        
        if recent_predictions.empty:
            logger.warning("No recent predictions for tuning")
            return {}
        
        # Analyze performance by market
        performance_by_market = recent_predictions.groupby('market').agg({
            'correct': 'mean',
            'profit': 'sum',
            'calibrated_probability': 'mean'
        })
        
        # Identify underperforming markets
        underperforming = performance_by_market[performance_by_market['correct'] < 0.55]
        
        if not underperforming.empty:
            logger.info(f"Retraining {len(underperforming)} underperforming markets")
            
            # Fetch fresh training data
            df = await self.db_manager.get_training_data(days=30)
            
            # Retrain only underperforming markets
            results = {}
            for market in underperforming.index:
                logger.info(f"Retraining market: {market}")
                training_datasets = self.prepare_training_data(df)
                
                if market in training_datasets:
                    X, y = training_datasets[market]
                    performance = self.ml_engine.train_market_model(X, y, market)
                    results[market] = performance
            
            # Save updated models
            self.ml_engine.save_models(self.models_dir)
            
            return results
        else:
            logger.info("All markets performing well, no tuning needed")
            return {}


async def main():
    """Main training pipeline"""
    logger.add("logs/training_{time}.log", rotation="500 MB")
    
    trainer = ModelTrainer()
    
    # Step 1: Backfill historical data
    logger.info("Step 1: Backfilling historical data...")
    df = await trainer.backfill_historical_data(days=settings.BACKFILL_DAYS)
    
    if df.empty:
        logger.error("No data available for training")
        return
    
    # Step 2: Split data for training and validation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Step 3: Train models
    logger.info("\nStep 2: Training models...")
    performance_results = await trainer.train_all_markets(train_df)
    
    # Step 4: Validate models
    logger.info("\nStep 3: Validating models...")
    validation_results = await trainer.validate_models(test_df)
    
    # Step 5: Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Markets trained: {len(performance_results)}")
    logger.info(f"Markets validated: {len(validation_results)}")
    logger.info(f"Models saved to: {trainer.models_dir}")
    
    # Display best markets
    best_markets = trainer.ml_engine.market_selector.select_best_markets(top_n=5)
    logger.info(f"\nTop 5 markets for prediction:")
    for i, market in enumerate(best_markets, 1):
        if market in performance_results:
            perf = performance_results[market]
            logger.info(f"  {i}. {market}: Accuracy={perf.accuracy:.3f}, AUC={perf.auc_roc:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
