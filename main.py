#!/usr/bin/env python3
"""
Main entry point for the AI Betting Predictor System
World-class production-grade betting prediction backend
"""

import asyncio
import argparse
import schedule
import time
from datetime import datetime
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from utils.database import DatabaseManager
from utils.api_client import APIFootballClient
from utils.telegram_bot import TelegramBot
from services.live_scanner import LiveMatchScanner
from services.self_learning import SelfLearningSystem
from services.predictor import PredictionService
from train_models import ModelTrainer

class BettingPredictionSystem:
    """Main system orchestrator"""
    
    def __init__(self, mode: str = 'live'):
        self.mode = mode
        self.db = DatabaseManager()
        self.api = APIFootballClient()
        self.telegram = TelegramBot()
        self.scanner = LiveMatchScanner(self.api, self.db)
        self.learner = SelfLearningSystem(self.db)
        self.predictor = PredictionService()
        self.trainer = ModelTrainer(self.db)
        
        # Ensure directories exist
        self.setup_directories()
        
        # Create database tables
        self.db.create_tables()
        
        logger.info(f"Betting Prediction System initialized in {mode} mode")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['data/models', 'data/processed', 'logs', 'exports']
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def run_live_mode(self):
        """Run in live prediction mode"""
        logger.info("Starting live prediction mode...")
        
        # Start continuous scanning
        scan_task = asyncio.create_task(self.scanner.continuous_scan())
        
        # Schedule daily tasks
        self.schedule_daily_tasks()
        
        # Start health monitoring
        health_task = asyncio.create_task(self.monitor_health())
        
        # Keep running
        try:
            await asyncio.gather(scan_task, health_task)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"System error: {e}")
            raise
    
    def run_training_mode(self):
        """Run in training mode"""
        logger.info("Starting training mode...")
        
        # Train models
        self.trainer.train_all_models()
        
        # Evaluate performance
        self.trainer.evaluate_models()
        
        # Optimize thresholds
        self.trainer.optimize_thresholds()
        
        logger.info("Training completed successfully")
    
    async def run_manual_control(self, command: str, **kwargs):
        """Run manual control commands"""
        
        commands = {
            'scan': self.manual_scan,
            'train': self.manual_train,
            'digest': self.generate_daily_digest,
            'tune': self.auto_tune,
            'backfill': self.backfill_data,
            'health': self.check_health,
            'analyze': self.analyze_performance
        }
        
        if command not in commands:
            logger.error(f"Unknown command: {command}")
            return
        
        logger.info(f"Executing manual command: {command}")
        
        if asyncio.iscoroutinefunction(commands[command]):
            await commands[command](**kwargs)
        else:
            commands[command](**kwargs)
    
    async def manual_scan(self, leagues: list = None):
        """Manual live scan"""
        predictions = await self.scanner.scan_live_matches(leagues)
        
        if predictions:
            for pred in predictions:
                await self.telegram.send_prediction(pred)
        
        return predictions
    
    def manual_train(self):
        """Manual model training"""
        self.run_training_mode()
    
    async def generate_daily_digest(self):
        """Generate daily performance digest"""
        from datetime import datetime, timedelta
        
        yesterday = datetime.now() - timedelta(days=1)
        
        # Get yesterday's predictions
        query = """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN br.is_correct THEN 1 ELSE 0 END) as correct_predictions,
            AVG(br.profit_loss) as avg_profit,
            p.prediction_type
        FROM predictions p
        LEFT JOIN bet_results br ON p.id = br.prediction_id
        WHERE DATE(p.created_at) = DATE(%s)
        GROUP BY p.prediction_type
        """
        
        results = self.db.execute_query(
            query, (yesterday.date(),), fetch=True
        )
        
        # Generate digest message
        digest = f"📊 Daily Digest for {yesterday.date()}\n\n"
        
        for row in results:
            accuracy = (row['correct_predictions'] / row['total_predictions'] * 100 
                       if row['total_predictions'] > 0 else 0)
            digest += (
                f"{row['prediction_type']}:\n"
                f"  Predictions: {row['total_predictions']}\n"
                f"  Accuracy: {accuracy:.1f}%\n"
                f"  Avg Profit: {row['avg_profit']:.2f}\n\n"
            )
        
        await self.telegram.send_message(digest)
        
        return digest
    
    async def auto_tune(self):
        """Auto-tune model parameters"""
        logger.info("Starting auto-tuning...")
        
        # Get recent data for tuning
        training_data = self.db.get_training_data(limit=5000)
        
        if len(training_data) < 1000:
            logger.warning("Insufficient data for auto-tuning")
            return
        
        # Perform hyperparameter optimization
        self.trainer.optimize_hyperparameters(training_data)
        
        logger.info("Auto-tuning completed")
    
    async def backfill_data(self, days: int = 30):
        """Backfill historical data"""
        logger.info(f"Backfilling data for last {days} days...")
        
        # Get historical matches
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # This would call API to get historical matches
        # Implementation depends on API limits and requirements
        
        logger.info("Backfilling completed")
    
    async def check_health(self):
        """Check system health"""
        health_status = {
            'database': self.check_database_health(),
            'api': await self.check_api_health(),
            'models': self.check_models_health(),
            'telegram': await self.check_telegram_health(),
            'disk_space': self.check_disk_space(),
            'memory': self.check_memory_usage()
        }
        
        # Log health status
        for component, status in health_status.items():
            if status['healthy']:
                logger.info(f"{component}: HEALTHY - {status.get('message', '')}")
            else:
                logger.warning(f"{component}: UNHEALTHY - {status.get('message', '')}")
        
        return health_status
    
    def check_database_health(self):
        """Check database connection and performance"""
        try:
            # Test connection
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
            
            # Check table sizes
            query = """
            SELECT 
                (SELECT COUNT(*) FROM predictions) as predictions_count,
                (SELECT COUNT(*) FROM bet_results) as results_count,
                (SELECT COUNT(*) FROM live_matches) as live_matches_count
            """
            
            counts = self.db.execute_query(query, fetch=True)[0]
            
            return {
                'healthy': True,
                'message': f"Connected. Counts: {counts}"
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Connection failed: {e}"
            }
    
    async def check_api_health(self):
        """Check API Football health"""
        try:
            # Make a simple API call
            response = await self.api.make_request('/status')
            return {
                'healthy': response.get('response', {}).get('account', {}).get('active', False),
                'message': f"Plan: {response.get('response', {}).get('account', {}).get('plan', 'unknown')}"
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"API error: {e}"
            }
    
    def check_models_health(self):
        """Check if models are loaded and functional"""
        try:
            models_dir = Path('data/models')
            models = list(models_dir.glob('*.joblib'))
            
            if not models:
                return {
                    'healthy': False,
                    'message': "No models found"
                }
            
            # Try to load each model
            for model_path in models:
                try:
                    import joblib
                    joblib.load(model_path)
                except:
                    return {
                        'healthy': False,
                        'message': f"Failed to load {model_path.name}"
                    }
            
            return {
                'healthy': True,
                'message': f"Loaded {len(models)} models"
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Model check error: {e}"
            }
    
    async def check_telegram_health(self):
        """Check Telegram bot health"""
        try:
            await self.telegram.send_message("🤖 Health check - System is running")
            return {
                'healthy': True,
                'message': "Bot active"
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Telegram error: {e}"
            }
    
    def check_disk_space(self):
        """Check disk space"""
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free // (2**30)
        free_percent = (free / total) * 100
        
        healthy = free_gb > 5 and free_percent > 10
        
        return {
            'healthy': healthy,
            'message': f"{free_gb}GB free ({free_percent:.1f}%)"
        }
    
    def check_memory_usage(self):
        """Check memory usage"""
        import psutil
        
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        
        healthy = used_percent < 85
        
        return {
            'healthy': healthy,
            'message': f"{used_percent:.1f}% used"
        }
    
    async def monitor_health(self, interval: int = 300):
        """Continuous health monitoring"""
        while True:
            try:
                health = await self.check_health()
                
                # Check if any component is unhealthy
                unhealthy = [c for c, s in health.items() if not s['healthy']]
                
                if unhealthy:
                    alert = f"🚨 Health Alert - Unhealthy components: {', '.join(unhealthy)}"
                    logger.warning(alert)
                    
                    # Send alert
                    await self.telegram.send_message(alert)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    def schedule_daily_tasks(self):
        """Schedule daily maintenance tasks"""
        
        # Daily digest at 8:00 AM
        schedule.every().day.at("08:00").do(
            lambda: asyncio.create_task(self.generate_daily_digest())
        )
        
        # Auto-tuning at 4:00 AM (low traffic)
        schedule.every().day.at("04:00").do(
            lambda: asyncio.create_task(self.auto_tune())
        )
        
        # Self-learning analysis at 2:00 AM
        schedule.every().day.at("02:00").do(
            lambda: self.learner.analyze_results(days_back=7)
        )
        
        # Database cleanup (keep 90 days)
        schedule.every().day.at("03:00").do(self.cleanup_database)
        
        logger.info("Daily tasks scheduled")
    
    def cleanup_database(self):
        """Clean up old data"""
        query = """
        DELETE FROM predictions 
        WHERE created_at < NOW() - INTERVAL '90 days'
        """
        
        self.db.execute_query(query)
        logger.info("Database cleanup completed")
    
    async def analyze_performance(self, days: int = 30):
        """Analyze system performance"""
        
        logger.info(f"Analyzing performance for last {days} days...")
        
        # Get performance metrics
        query = """
        SELECT 
            DATE(p.created_at) as date,
            p.prediction_type,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN br.is_correct THEN 1 ELSE 0 END) as correct_predictions,
            AVG(p.confidence) as avg_confidence,
            AVG(br.profit_loss) as avg_profit
        FROM predictions p
        LEFT JOIN bet_results br ON p.id = br.prediction_id
        WHERE p.created_at >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(p.created_at), p.prediction_type
        ORDER BY date DESC, p.prediction_type
        """
        
        performance_data = self.db.execute_query(
            query, (days,), fetch=True
        )
        
        # Generate analysis report
        report = self.generate_performance_report(performance_data)
        
        # Save report
        report_path = f"exports/performance_report_{datetime.now().date()}.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance analysis saved to {report_path}")
        return report
    
    def generate_performance_report(self, data):
        """Generate performance report from data"""
        
        # This would create a comprehensive performance analysis
        # Implementation depends on your specific requirements
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'data_points': len(data),
            'summary': {},
            'trends': {},
            'recommendations': []
        }
        
        return report

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="AI Betting Prediction System")
    parser.add_argument('--mode', choices=['live', 'train', 'manual'], 
                       default='live', help='Operation mode')
    parser.add_argument('--command', help='Manual command to execute')
    parser.add_argument('--leagues', nargs='+', type=int, 
                       help='League IDs for manual scan')
    parser.add_argument('--days', type=int, default=7, 
                       help='Days for backfill/analysis')
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required environment variables
    required_vars = ['API_FOOTBALL_KEY', 'TELEGRAM_BOT_TOKEN', 'DATABASE_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        sys.exit(1)
    
    # Create and run system
    system = BettingPredictionSystem(mode=args.mode)
    
    try:
        if args.mode == 'live':
            asyncio.run(system.run_live_mode())
        elif args.mode == 'train':
            system.run_training_mode()
        elif args.mode == 'manual' and args.command:
            asyncio.run(system.run_manual_control(
                args.command, 
                leagues=args.leagues,
                days=args.days
            ))
        else:
            logger.error("Manual mode requires --command argument")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
