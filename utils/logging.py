import logging
import os
from typing import Optional

class CustomFormatter(logging.Formatter):
    """Custom log formatter with job ID support"""
    
    def format(self, record):
        if not hasattr(record, 'job_id'):
            record.job_id = 'main'
        return super().format(record)

def setup_logging():
    """Setup application logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    handler = logging.StreamHandler()
    formatter = CustomFormatter(
        "[%(asctime)s] %(levelname)s [%(job_id)s] - %(message)s"
    )
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level))
    
    # Configure our application logger
    app_logger = logging.getLogger("goalsniper")
    app_logger.handlers = [handler]
    app_logger.setLevel(getattr(logging, log_level))
    app_logger.propagate = False

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    logger = logging.getLogger(f"goalsniper.{name}")
    
    # Ensure job_id attribute is available
    def _log_with_job_id(msg, *args, job_id: Optional[str] = None, **kwargs):
        if job_id:
            extra = kwargs.get('extra', {})
            extra['job_id'] = job_id
            kwargs['extra'] = extra
        logger.info(msg, *args, **kwargs)
    
    logger.log_with_job_id = _log_with_job_id
    return logger
