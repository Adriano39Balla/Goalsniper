#!/usr/bin/env python3
"""
Market-Agnostic Prediction Engine - Main Application
Robi_Superbrain v10+
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis

from engine.event_engine import EventProbabilityEngine
from engine.market_generator import MarketOpportunityGenerator
from engine.value_filter import ValueConfidenceFilter
from engine.learning_loop import LearningFeedbackLoop
from engine.database import SessionLocal, init_db, get_db
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
event_engine: Optional[EventProbabilityEngine] = None
market_generator: Optional[MarketOpportunityGenerator] = None
value_filter: Optional[ValueConfidenceFilter] = None
learning_loop: Optional[LearningFeedbackLoop] = None
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for startup/shutdown events
    """
    # Startup
    logger.info("Starting Market-Agnostic Prediction Engine...")
    
    # Initialize Redis
    global redis_client
    redis_client = redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Initialize database
    await init_db()
    
    # Initialize engines
    global event_engine, market_generator, value_filter, learning_loop
    event_engine = EventProbabilityEngine(redis_client=redis_client)
    market_generator = MarketOpportunityGenerator()
    value_filter = ValueConfidenceFilter()
    learning_loop = LearningFeedbackLoop()
    
    # Warm up models
    logger.info("Warming up prediction models...")
    await event_engine.warm_up()
    
    # Start background tasks
    background_tasks = BackgroundTasks()
    background_tasks.add_task(periodic_model_update)
    background_tasks.add_task(cleanup_old_data)
    
    logger.info("Engine started successfully!")
    yield
    
    # Shutdown
    logger.info("Shutting down engine...")
    if redis_client:
        await redis_client.close()
    logger.info("Engine shutdown complete.")

# Create FastAPI app
app = FastAPI(
    title="Market-Agnostic Prediction Engine",
    description="Robi_Superbrain v10+ - Discovers value instead of being told what to predict",
    version="10.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class MatchState(BaseModel):
    """Input match state from API-Football"""
    match_id: str
    minute: int
    home_score: int = 0
    away_score: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
    shots_on_target: Dict[str, int] = Field(default_factory=dict)
    shots_off_target: Dict[str, int] = Field(default_factory=dict)
    dangerous_attacks: Dict[str, int] = Field(default_factory=dict)
    corners: Dict[str, int] = Field(default_factory=dict)
    possession: Dict[str, float] = Field(default_factory=dict)
    yellow_cards: Dict[str, int] = Field(default_factory=dict)
    red_cards: Dict[str, int] = Field(default_factory=dict)
    league_id: str
    league_strength: float = 1.0
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)

class PredictionRequest(BaseModel):
    """Request for predictions"""
    match_state: MatchState
    markets_to_consider: Optional[List[str]] = None
    confidence_threshold: float = 0.6
    max_recommendations: int = 3

class PredictionResponse(BaseModel):
    """Response with market recommendations"""
    match_id: str
    minute: int
    game_state: str
    recommendations: List[Dict[str, Any]]
    event_probabilities: Dict[str, float]
    confidence_scores: Dict[str, float]
    warnings: List[str] = Field(default_factory=list)

class LearningUpdate(BaseModel):
    """Feedback for learning loop"""
    match_id: str
    market: str
    outcome: bool  # True if prediction was correct
    actual_result: Optional[Dict[str, Any]] = None
    confidence_at_prediction: float
    match_state_at_prediction: Dict[str, Any]

# Background Tasks
async def periodic_model_update():
    """Periodically update models with new data"""
    while True:
        try:
            logger.info("Starting periodic model update...")
            # This would fetch new data and retrain models
            # For now, just a placeholder
            await asyncio.sleep(settings.MODEL_UPDATE_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic model update: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

async def cleanup_old_data():
    """Clean up old cache and temporary data"""
    while True:
        try:
            if redis_client:
                # Remove predictions older than 24 hours
                await redis_client.delete_old_predictions(hours=24)
            await asyncio.sleep(3600)  # Run hourly
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            await asyncio.sleep(300)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "10.0.0",
        "components": {
            "event_engine": event_engine is not None,
            "market_generator": market_generator is not None,
            "value_filter": value_filter is not None,
            "learning_loop": learning_loop is not None,
            "redis": redis_client is not None,
            "database": True  # Simplified check
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_markets(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """
    Main prediction endpoint - analyzes match state and recommends markets
    """
    try:
        logger.info(f"Predicting markets for match {request.match_state.match_id}")
        
        # Step 1: Calculate event probabilities
        event_probs = await event_engine.predict_events(
            request.match_state.dict(),
            db=db
        )
        
        # Step 2: Generate market opportunities
        market_candidates = market_generator.find_opportunities(
            event_probs,
            markets_to_consider=request.markets_to_consider
        )
        
        # Step 3: Filter by value and confidence
        scored_markets = []
        for candidate in market_candidates:
            scored = value_filter.score_market(
                candidate,
                request.match_state.dict(),
                db=db
            )
            if scored['final_score'] >= request.confidence_threshold:
                scored_markets.append(scored)
        
        # Step 4: Apply game-state intelligence and get final selection
        game_state = event_engine.classify_game_state(request.match_state.dict())
        final_recommendations = value_filter.select_best_opportunities(
            scored_markets,
            game_state,
            max_count=request.max_recommendations
        )
        
        # Step 5: Log the prediction for learning
        background_tasks.add_task(
            log_prediction_for_learning,
            request.match_state.match_id,
            final_recommendations,
            event_probs,
            game_state
        )
        
        # Prepare warnings if any
        warnings = []
        if len(final_recommendations) == 0:
            warnings.append("No high-confidence opportunities found")
        
        # Check for potential traps
        trap_warnings = event_engine.detect_traps(event_probs)
        warnings.extend(trap_warnings)
        
        return PredictionResponse(
            match_id=request.match_state.match_id,
            minute=request.match_state.minute,
            game_state=game_state,
            recommendations=final_recommendations,
            event_probabilities={
                k: v for k, v in event_probs.items() 
                if isinstance(v, (int, float))
            },
            confidence_scores={
                rec['market']: rec['final_score'] 
                for rec in final_recommendations
            },
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn")
async def submit_learning_update(
    update: LearningUpdate,
    db=Depends(get_db)
):
    """
    Submit feedback for the learning loop
    """
    try:
        await learning_loop.process_outcome(
            prediction=update.dict(),
            db=db
        )
        return {"status": "learning_update_received"}
    except Exception as e:
        logger.error(f"Learning update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-performance")
async def get_market_performance(
    market: Optional[str] = None,
    league: Optional[str] = None,
    days: int = 30,
    db=Depends(get_db)
):
    """
    Get performance metrics for markets
    """
    try:
        performance = await learning_loop.get_performance_stats(
            market=market,
            league=league,
            days=days,
            db=db
        )
        return performance
    except Exception as e:
        logger.error(f"Performance query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status")
async def get_model_status():
    """
    Get status of all models
    """
    try:
        status = {
            "goal_model": event_engine.event_models['goal'].get_status(),
            "corner_model": event_engine.event_models['corner'].get_status(),
            "card_model": event_engine.event_models['card'].get_status(),
            "state_classifier": event_engine.state_classifier.get_status(),
            "last_trained": await redis_client.get("last_model_training"),
            "cache_stats": await redis_client.info("stats")
        }
        return status
    except Exception as e:
        logger.error(f"Model status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper Functions
async def log_prediction_for_learning(
    match_id: str,
    recommendations: List[Dict],
    event_probs: Dict,
    game_state: str
):
    """Log prediction for future learning"""
    try:
        for rec in recommendations:
            await learning_loop.log_prediction(
                match_id=match_id,
                market=rec['market'],
                confidence=rec['final_score'],
                event_probabilities=event_probs,
                game_state=game_state,
                timestamp=datetime.utcnow()
            )
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

# Signal Handlers
def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
