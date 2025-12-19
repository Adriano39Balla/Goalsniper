"""
Main Application - FastAPI Control System
World-class betting predictions backend with comprehensive manual control
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from ml_engine import GradientBoostingEnsemble, PredictionResult
from data_pipeline import FootballDataPipeline, LiveMatchData
from database import DatabaseManager
from telegram_notifier import TelegramNotifier
from train_models import ModelTrainer


# Initialize FastAPI app
app = FastAPI(
    title="AI Betting Predictions Backend",
    description="Production-grade betting predictions with ML and live in-play focus",
    version="1.0.0"
)

# Global instances
ml_engine = GradientBoostingEnsemble(model_type=settings.MODEL_TYPE)
data_pipeline = FootballDataPipeline()
db_manager = DatabaseManager()
telegram_notifier = TelegramNotifier()
model_trainer = ModelTrainer()
scheduler = AsyncIOScheduler()

# System state
system_state = {
    'live_scan_active': False,
    'auto_tune_active': False,
    'last_scan_time': None,
    'predictions_today': 0,
    'models_loaded': False
}


# Pydantic models for API
class PredictionResponse(BaseModel):
    fixture_id: int
    home_team: str
    away_team: str
    market: str
    prediction: str
    probability: float
    confidence: float
    expected_value: float
    timestamp: str


class SystemStatus(BaseModel):
    status: str
    live_scan_active: bool
    auto_tune_active: bool
    models_loaded: bool
    last_scan_time: Optional[str]
    predictions_today: int
    uptime_seconds: float


class TrainingRequest(BaseModel):
    backfill_days: int = 90
    markets: Optional[List[str]] = None


class ManualPredictionRequest(BaseModel):
    fixture_id: int


# Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key for protected endpoints"""
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.add("logs/main_{time}.log", rotation="500 MB", retention="30 days")
    logger.info("Starting AI Betting Predictions Backend...")
    
    # Initialize database
    await db_manager.initialize_schema()
    
    # Load trained models
    models_dir = Path("models")
    if models_dir.exists():
        ml_engine.load_models(models_dir)
        system_state['models_loaded'] = True
        logger.info("Models loaded successfully")
    else:
        logger.warning("No trained models found. Please run training first.")
    
    # Start scheduler
    scheduler.start()
    logger.info("Scheduler started")
    
    # Schedule daily digest
    scheduler.add_job(
        send_daily_digest,
        'cron',
        hour=8,
        minute=0,
        id='daily_digest'
    )
    
    logger.info("System startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down system...")
    
    # Stop live scan if active
    if system_state['live_scan_active']:
        system_state['live_scan_active'] = False
    
    # Close connections
    await data_pipeline.close()
    db_manager.close()
    
    # Shutdown scheduler
    scheduler.shutdown()
    
    logger.info("System shutdown complete")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": system_state['models_loaded'],
        "live_scan_active": system_state['live_scan_active']
    }


# System status endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    import time
    
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return SystemStatus(
        status="operational" if system_state['models_loaded'] else "no_models",
        live_scan_active=system_state['live_scan_active'],
        auto_tune_active=system_state['auto_tune_active'],
        models_loaded=system_state['models_loaded'],
        last_scan_time=system_state['last_scan_time'],
        predictions_today=system_state['predictions_today'],
        uptime_seconds=uptime
    )


# Live scan control
@app.post("/api/scan/start")
async def start_live_scan(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Start live match scanning and prediction generation"""
    if system_state['live_scan_active']:
        raise HTTPException(status_code=400, detail="Live scan already active")
    
    if not system_state['models_loaded']:
        raise HTTPException(status_code=400, detail="Models not loaded. Please train models first.")
    
    system_state['live_scan_active'] = True
    background_tasks.add_task(live_scan_loop)
    
    logger.info("Live scan started")
    await db_manager.log_system_event("INFO", "Live scan started", "main")
    
    return {"message": "Live scan started", "status": "active"}


@app.post("/api/scan/stop")
async def stop_live_scan(api_key: str = Depends(verify_api_key)):
    """Stop live match scanning"""
    if not system_state['live_scan_active']:
        raise HTTPException(status_code=400, detail="Live scan not active")
    
    system_state['live_scan_active'] = False
    
    logger.info("Live scan stopped")
    await db_manager.log_system_event("INFO", "Live scan stopped", "main")
    
    return {"message": "Live scan stopped", "status": "inactive"}


# Training control
@app.post("/api/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Start model training with historical data"""
    background_tasks.add_task(run_training, request.backfill_days)
    
    logger.info(f"Training started with {request.backfill_days} days backfill")
    await db_manager.log_system_event("INFO", f"Training started ({request.backfill_days} days)", "training")
    
    return {
        "message": "Training started",
        "backfill_days": request.backfill_days,
        "status": "running"
    }


@app.get("/api/training/status")
async def get_training_status():
    """Get training status and history"""
    # Get recent training history from database
    with db_manager.get_connection() as conn:
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT * FROM training_history ORDER BY timestamp DESC LIMIT 10",
            conn
        )
    
    return {
        "recent_training": df.to_dict('records') if not df.empty else [],
        "models_loaded": system_state['models_loaded']
    }


# Auto-tune control
@app.post("/api/autotune/start")
async def start_auto_tune(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Start automatic model tuning based on performance"""
    if system_state['auto_tune_active']:
        raise HTTPException(status_code=400, detail="Auto-tune already active")
    
    system_state['auto_tune_active'] = True
    
    # Schedule auto-tune to run every 24 hours
    scheduler.add_job(
        run_auto_tune,
        'interval',
        hours=settings.RETRAIN_INTERVAL_HOURS,
        id='auto_tune'
    )
    
    logger.info("Auto-tune started")
    await db_manager.log_system_event("INFO", "Auto-tune started", "main")
    
    return {"message": "Auto-tune started", "interval_hours": settings.RETRAIN_INTERVAL_HOURS}


@app.post("/api/autotune/stop")
async def stop_auto_tune(api_key: str = Depends(verify_api_key)):
    """Stop automatic model tuning"""
    if not system_state['auto_tune_active']:
        raise HTTPException(status_code=400, detail="Auto-tune not active")
    
    system_state['auto_tune_active'] = False
    
    try:
        scheduler.remove_job('auto_tune')
    except:
        pass
    
    logger.info("Auto-tune stopped")
    await db_manager.log_system_event("INFO", "Auto-tune stopped", "main")
    
    return {"message": "Auto-tune stopped"}


# Backfill control
@app.post("/api/backfill/start")
async def start_backfill(
    days: int,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Backfill historical match data"""
    background_tasks.add_task(run_backfill, days)
    
    logger.info(f"Backfill started for {days} days")
    await db_manager.log_system_event("INFO", f"Backfill started ({days} days)", "backfill")
    
    return {"message": f"Backfill started for {days} days", "status": "running"}


# Daily digest control
@app.post("/api/digest/send")
async def send_daily_digest_now(api_key: str = Depends(verify_api_key)):
    """Send daily digest immediately"""
    await send_daily_digest()
    return {"message": "Daily digest sent"}


# Manual prediction
@app.post("/api/predict/manual", response_model=List[PredictionResponse])
async def manual_prediction(
    request: ManualPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate predictions for a specific fixture manually"""
    if not system_state['models_loaded']:
        raise HTTPException(status_code=400, detail="Models not loaded")
    
    # Fetch fixture data
    matches = await data_pipeline.fetch_live_matches()
    target_match = None
    
    for match in matches:
        if match.fixture_id == request.fixture_id:
            target_match = match
            break
    
    if not target_match:
        raise HTTPException(status_code=404, detail="Fixture not found or not live")
    
    # Generate predictions
    predictions = await generate_predictions_for_match(target_match)
    
    # Format response
    responses = []
    for pred in predictions:
        responses.append(PredictionResponse(
            fixture_id=pred.fixture_id,
            home_team=target_match.home_team,
            away_team=target_match.away_team,
            market=pred.market,
            prediction=pred.prediction,
            probability=pred.calibrated_probability,
            confidence=pred.confidence_score,
            expected_value=pred.expected_value,
            timestamp=pred.timestamp.isoformat()
        ))
    
    return responses


# Statistics and analytics
@app.get("/api/statistics/dashboard")
async def get_dashboard_statistics():
    """Get comprehensive dashboard statistics"""
    stats = await db_manager.get_statistics_dashboard()
    return stats


@app.get("/api/statistics/market/{market}")
async def get_market_statistics(market: str, days: int = 30):
    """Get statistics for a specific market"""
    perf = await db_manager.get_market_performance(market, days)
    
    if not perf:
        raise HTTPException(status_code=404, detail="Market not found")
    
    return perf


@app.get("/api/predictions/recent")
async def get_recent_predictions(limit: int = 50):
    """Get recent predictions"""
    df = await db_manager.get_recent_predictions(days=7)
    
    if df.empty:
        return []
    
    return df.head(limit).to_dict('records')


# Background tasks
async def live_scan_loop():
    """Main live scanning loop"""
    logger.info("Live scan loop started")
    
    while system_state['live_scan_active']:
        try:
            # Fetch live matches
            matches = await data_pipeline.fetch_live_matches()
            
            if matches:
                logger.info(f"Processing {len(matches)} live matches")
                
                for match in matches:
                    # Save match data
                    await db_manager.save_live_match(match.to_dict())
                    
                    # Generate predictions
                    predictions = await generate_predictions_for_match(match)
                    
                    # Send high-confidence predictions to Telegram
                    for pred in predictions:
                        if pred.confidence_score >= settings.MIN_CONFIDENCE_THRESHOLD:
                            await telegram_notifier.send_prediction(
                                match=match,
                                prediction=pred
                            )
                            system_state['predictions_today'] += 1
            
            system_state['last_scan_time'] = datetime.now().isoformat()
            
            # Wait before next scan (30 seconds for live updates)
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in live scan loop: {e}")
            await asyncio.sleep(30)
    
    logger.info("Live scan loop stopped")


async def generate_predictions_for_match(match: LiveMatchData) -> List[PredictionResult]:
    """Generate predictions for a match"""
    import pandas as pd
    
    predictions = []
    
    # Convert match data to DataFrame
    match_dict = match.to_dict()
    df = pd.DataFrame([match_dict])
    
    # Engineer features
    df_engineered = ml_engine.engineer_features(df)
    
    # Get best markets to predict
    best_markets = ml_engine.market_selector.select_best_markets(top_n=3)
    
    for market in best_markets:
        if market not in ml_engine.models:
            continue
        
        try:
            # Get features for this market
            if market in ml_engine.market_selector.market_features:
                feature_cols = ml_engine.market_selector.market_features[market]
                X = df_engineered[feature_cols]
            else:
                X = df_engineered.select_dtypes(include=['number'])
            
            # Generate prediction
            pred = ml_engine.predict(X, match.fixture_id, market)
            
            if pred and pred.confidence_score >= settings.MIN_CONFIDENCE_THRESHOLD:
                # Save to database
                pred_dict = {
                    'fixture_id': pred.fixture_id,
                    'market': pred.market,
                    'prediction': pred.prediction,
                    'probability': pred.probability,
                    'calibrated_probability': pred.calibrated_probability,
                    'confidence_score': pred.confidence_score,
                    'expected_value': pred.expected_value,
                    'model_version': pred.model_version,
                    'features_used': json.dumps(pred.features_used),
                    'timestamp': pred.timestamp
                }
                await db_manager.save_prediction(pred_dict)
                
                predictions.append(pred)
                
        except Exception as e:
            logger.error(f"Error generating prediction for market {market}: {e}")
    
    return predictions


async def run_training(backfill_days: int):
    """Run model training in background"""
    try:
        logger.info(f"Starting training with {backfill_days} days backfill")
        
        # Backfill data
        df = await model_trainer.backfill_historical_data(days=backfill_days)
        
        if not df.empty:
            # Train models
            performance = await model_trainer.train_all_markets(df)
            
            # Reload models
            ml_engine.load_models(Path("models"))
            system_state['models_loaded'] = True
            
            logger.info("Training completed successfully")
            await db_manager.log_system_event("INFO", "Training completed", "training")
        else:
            logger.error("No data available for training")
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        await db_manager.log_system_event("ERROR", f"Training failed: {e}", "training")


async def run_auto_tune():
    """Run automatic model tuning"""
    try:
        logger.info("Starting auto-tune")
        
        results = await model_trainer.auto_tune_models()
        
        if results:
            # Reload models
            ml_engine.load_models(Path("models"))
            logger.info(f"Auto-tune completed, retrained {len(results)} markets")
        else:
            logger.info("Auto-tune: No retraining needed")
            
    except Exception as e:
        logger.error(f"Auto-tune error: {e}")


async def run_backfill(days: int):
    """Run backfill in background"""
    try:
        logger.info(f"Starting backfill for {days} days")
        
        df = await model_trainer.backfill_historical_data(days=days)
        
        logger.info(f"Backfill completed: {len(df)} records")
        await db_manager.log_system_event("INFO", f"Backfill completed ({len(df)} records)", "backfill")
        
    except Exception as e:
        logger.error(f"Backfill error: {e}")
        await db_manager.log_system_event("ERROR", f"Backfill failed: {e}", "backfill")


async def send_daily_digest():
    """Send daily performance digest"""
    try:
        stats = await db_manager.get_statistics_dashboard()
        await telegram_notifier.send_daily_digest(stats)
        logger.info("Daily digest sent")
        
    except Exception as e:
        logger.error(f"Error sending daily digest: {e}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Betting Predictions Backend",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "status": "/api/status",
            "scan_start": "/api/scan/start",
            "scan_stop": "/api/scan/stop",
            "training": "/api/training/start",
            "autotune": "/api/autotune/start",
            "backfill": "/api/backfill/start",
            "statistics": "/api/statistics/dashboard"
        }
    }


if __name__ == "__main__":
    import uvicorn
    import time
    
    app.state.start_time = time.time()
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )
