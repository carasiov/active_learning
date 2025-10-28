"""Structured logging configuration for dashboard."""

import logging
import sys
from pathlib import Path
from typing import Optional


class DashboardLogger:
    """Centralized logging configuration for dashboard components."""
    
    _initialized = False
    _log_file = "/tmp/ssvae_dashboard.log"
    
    @classmethod
    def setup(cls, log_file: Optional[str] = None, console_level: int = logging.WARNING):
        """Configure structured logging for dashboard.
        
        Args:
            log_file: Path to log file (default: /tmp/ssvae_dashboard.log)
            console_level: Console log level (default: WARNING, use DEBUG to see everything)
        """
        if cls._initialized:
            return
        
        if log_file:
            cls._log_file = log_file
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler - detailed logs
        file_handler = logging.FileHandler(cls._log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler - configurable level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(simple_formatter)
        
        # Configure dashboard loggers
        logger_names = [
            'dashboard.state',
            'dashboard.callbacks',
            'dashboard.commands',
            'dashboard.model_manager',
            'dashboard.app',
        ]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.handlers = []
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.propagate = False
        
        # Log startup
        app_logger = logging.getLogger('dashboard.app')
        app_logger.info("="*60)
        app_logger.info("Dashboard logging initialized")
        app_logger.info(f"Log file: {cls._log_file}")
        app_logger.info(f"Console level: {logging.getLevelName(console_level)}")
        app_logger.info("="*60)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, component: str) -> logging.Logger:
        """Get logger for a specific component.
        
        Args:
            component: Component name (e.g., 'callbacks', 'commands', 'state')
        
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        return logging.getLogger(f'dashboard.{component}')


def get_logger(component: str) -> logging.Logger:
    """Convenience function to get a logger.
    
    Args:
        component: Component name (e.g., 'callbacks', 'commands', 'state')
    
    Returns:
        Configured logger instance
    """
    return DashboardLogger.get_logger(component)
