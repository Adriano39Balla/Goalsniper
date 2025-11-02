import logging
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime

from core.config import config
from core.database import db
from services.telegram import telegram_service
from jobs.scanners import production_scanner

log = logging.getLogger("goalsniper.scheduler")

class JobScheduler:
    """Job scheduling management"""
    
    def __init__(self):
        self.scheduler: Optional[BackgroundScheduler] = None
        self.started = False
    
    def start(self):
        """Start the scheduler"""
        if self.started or not config.scheduler.enabled:
            return
        
        try:
            self.scheduler = BackgroundScheduler(timezone="UTC")
            self._setup_jobs()
            self.scheduler.start()
            self.started = True
            
            telegram_service.send_system_message(
                "🚀 goalsniper AI mode (in-play) with ENHANCED PREDICTIONS started."
            )
            log.info("[SCHEDULER] Scheduler started successfully")
            
        except Exception as e:
            log.exception("[SCHEDULER] Failed to start scheduler: %s", e)
    
    def _setup_jobs(self):
        """Setup all scheduled jobs"""
        # Production scanning
        self.scheduler.add_job(
            lambda: self._run_with_lock(1001, production_scanner.scan),
            "interval",
            seconds=config.scheduler.scan_interval_sec,
            id="scan",
            max_instances=1,
            coalesce=True
        )
        
        # Backfill results
        self.scheduler.add_job(
            lambda: self._run_with_lock(1002, self._backfill_results),
            "interval",
            minutes=config.scheduler.backfill_every_min,
            id="backfill",
            max_instances=1,
            coalesce=True
        )
        
        # Model training
        if config.train_enable:
            self.scheduler.add_job(
                lambda: self._run_with_lock(1003, self._train_models),
                CronTrigger(
                    hour=config.scheduler.train_hour_utc,
                    minute=config.scheduler.train_minute_utc,
                    timezone="UTC"
                ),
                id="train",
                max_instances=1,
                coalesce=True
            )
        
        # Add other jobs (digest, auto-tune, retry, cache cleanup)
        # [Your existing job setup logic]
    
    def _run_with_lock(self, lock_key: int, job_function, *args, **kwargs):
        """Run job with database lock to prevent overlaps"""
        try:
            with db.get_cursor() as c:
                # Try to acquire advisory lock
                c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
                lock_acquired = c.fetchone()[0]
                
                if not lock_acquired:
                    log.info("[LOCK %s] Job already running, skipping", lock_key)
                    return None
                
                try:
                    return job_function(*args, **kwargs)
                finally:
                    # Release the lock
                    c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                    
        except Exception as e:
            log.exception("[LOCK %s] Job failed: %s", lock_key, e)
            return None
    
    def _backfill_results(self):
        """Backfill match results"""
        from services.maintenance import MaintenanceService
        maintenance = MaintenanceService()
        return maintenance.backfill_results()
    
    def _train_models(self):
        """Train models job"""
        from services.training import TrainingService
        trainer = TrainingService()
        return trainer.train_models()
    
    def shutdown(self):
        """Shutdown the scheduler"""
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
                self.started = False
                log.info("[SCHEDULER] Scheduler shut down")
            except Exception as e:
                log.warning("[SCHEDULER] Error shutting down scheduler: %s", e)
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self.started and self.scheduler is not None

# Global scheduler instance
job_scheduler = JobScheduler()
