"""
Logging utilities.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 10,
    backup_count: int = 5,
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
) -> None:
    """Set up logging with loguru."""
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format=log_format,
        level=level.upper(),
        colorize=True,
    )
    
    # Add file logger if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logger.add(
            log_file,
            format=log_format,
            level=level.upper(),
            rotation=f"{max_size_mb} MB",
            retention=backup_count,
            compression="zip",
        )
    
    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Setup logging for libraries using standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set up specific loggers
    for module in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging.getLogger(module).handlers = [InterceptHandler()]
        logging.getLogger(module).propagate = False


def log_request(route: str, payload: Dict[Any, Any]) -> None:
    """Log an API request."""
    logger.info(f"Request to {route}: {payload}")


def log_response(route: str, status_code: int, duration_ms: float) -> None:
    """Log an API response."""
    logger.info(f"Response from {route}: status_code={status_code}, duration={duration_ms:.2f}ms")


def log_model_load(model_id: str, quantization: str, device: str) -> None:
    """Log model loading."""
    logger.info(f"Loading model: {model_id} (quantization={quantization}, device={device})")


def log_model_unload(model_id: str) -> None:
    """Log model unloading."""
    logger.info(f"Unloading model: {model_id}")


def log_exception(e: Exception, context: str = "") -> None:
    """Log an exception."""
    if context:
        logger.exception(f"Exception in {context}: {str(e)}")
    else:
        logger.exception(f"Exception: {str(e)}")