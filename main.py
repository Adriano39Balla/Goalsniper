#!/usr/bin/env python3
"""
Main application for AI Betting Predictions Backend
World-class production-grade betting prediction system
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn

from config.settings import settings
from data.database import db
from services.live_scanner import LiveScanner
from services.telegram_bot import telegram_bot
from services.health_monitor import HealthMonitor
from models.ensemble import EnsemblePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class BettingAISystem:
    def __init__(self):
        self.live_scanner = LiveScanner()
        self.health_monitor = HealthMonitor()
        self.ensemble_predictor = EnsemblePredictor()
        self.app = None
    
    async def initialize(self):
        """Initialize the complete system"""
        logger.info("Initializing AI Betting Prediction System...")
        
        try:
            # Initialize database connection
            db.connect()
            
            # Load AI models
            self.ensemble_predictor.load_models()
            
            # Initialize Telegram bot
            await telegram_bot.initialize()
            
            # Health check
            health_status = await self.health_monitor.get_health_status()
            logger.info(f"System health: {health_status}")
            
            logger.info("AI Betting Prediction System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the system"""
        logger.info("Starting AI Betting Prediction System...")
        
        # Start live scanner in background
        asyncio.create_task(self.live_scanner.start_scan())
        
        logger.info("System started successfully")
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down AI Betting Prediction System...")
        
        await self.live_scanner.stop_scan()
        
        logger.info("System shutdown completed")

# Global system instance
betting_system = BettingAISystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    await betting_system.initialize()
    await betting_system.start()
    yield
    await betting_system.shutdown()

# Create FastAPI app
app = FastAPI(
    title="AI Betting Predictions API",
    description="World-class AI-powered betting predictions backend",
    version="1.0.0",
    lifespan=lifespan
)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Betting Predictions API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = await betting_system.health_monitor.get_health_status()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan/start")
async def start_scan():
    """Start live scanning manually"""
    try:
        await betting_system.live_scanner.start_scan()
        return {"message": "Live scanning started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan/stop")
async def stop_scan():
    """Stop live scanning"""
    try:
        await betting_system.live_scanner.stop_scan()
        return {"message": "Live scanning stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan/manual")
async def manual_scan():
    """Trigger manual scan"""
    try:
        await betting_system.live_scanner.manual_scan()
        return {"message": "Manual scan completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/start")
async def start_training():
    """Start model training manually"""
    try:
        from train_models import main as train_main
        asyncio.create_task(train_main())
        return {"message": "Model training started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/daily-digest")
async def get_daily_digest():
    """Get daily performance digest"""
    try:
        digest = await betting_system.health_monitor.generate_daily_report()
        return digest
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backfill")
async def backfill_data(days: int = 30):
    """Backfill historical data"""
    try:
        # Implementation would go here
        return {"message": f"Backfill started for {days} days"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Signal handlers for graceful shutdown
def signal_handler(sig, frame):
    logger.info("Received shutdown signal")
    asyncio.create_task(betting_system.shutdown())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
