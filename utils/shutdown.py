import logging
import signal
import atexit
import sys
from typing import Callable

log = logging.getLogger("goalsniper.shutdown")

class ShutdownManager:
    """Graceful shutdown management"""
    
    def __init__(self):
        self.shutdown_requested = False
        self.shutdown_handlers = []
        self.handlers_registered = False
    
    def register_shutdown_handler(self, handler: Callable):
        """Register a shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    def request_shutdown(self):
        """Request application shutdown"""
        if self.shutdown_requested:
            return
        
        self.shutdown_requested = True
        log.info("Shutdown requested, running cleanup handlers...")
        
        # Run all shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                log.warning("Shutdown handler error: %s", e)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self.shutdown_requested
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        if self.handlers_registered:
            return
        
        def signal_handler(signum, frame):
            log.info("Received signal %s, initiating shutdown...", signum)
            self.request_shutdown()
            sys.exit(0)
        
        def atexit_handler():
            if not self.shutdown_requested:
                log.info("Application exiting, running shutdown handlers...")
                self.request_shutdown()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register atexit handler
        atexit.register(atexit_handler)
        
        self.handlers_registered = True
        log.info("Shutdown handlers registered")

# Global shutdown manager instance
shutdown_manager = ShutdownManager()
