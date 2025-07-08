import sys
import json
from loguru import logger
from typing import Optional, Dict, Any
from pathlib import Path


class LoggingConfig:
    """Enhanced logging configuration with structured logging support."""
    
    @staticmethod
    def setup_logging(
        level: str = "INFO",
        log_file: Optional[str] = None,
        enable_json: bool = False,
        enable_file_logging: bool = False,
        max_file_size: str = "10 MB",
        retention_days: int = 7,
        app_name: str = "MMCT"
    ):
        """
        Configure logging with console and optional file output.
        
        Args:
            level: Log level
            log_file: Path to log file
            enable_json: Enable JSON format logging
            enable_file_logging: Enable file logging
            max_file_size: Maximum file size before rotation
            retention_days: Number of days to keep log files
            app_name: Application name for log context
        """
        # Remove default handler
        logger.remove()
        
        # Console handler format
        if enable_json:
            console_format = _json_formatter
        else:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        # Add console handler
        logger.add(
            sys.stdout,
            format=console_format,
            level=level,
            colorize=not enable_json,
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        # Add file handler if enabled
        if enable_file_logging:
            log_file_path = log_file or f"logs/{app_name.lower()}.log"
            
            # Ensure log directory exists
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            file_format = _json_formatter if enable_json else (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
            )
            
            logger.add(
                log_file_path,
                format=file_format,
                level=level,
                rotation=max_file_size,
                retention=f"{retention_days} days",
                compression="zip",
                enqueue=True,
                backtrace=True,
                diagnose=True
            )
        
        # Add request ID context
        logger.configure(
            extra={"app_name": app_name, "version": "1.0.0"}
        )
        
        logger.info(f"Logging initialized - Level: {level}, JSON: {enable_json}, File: {enable_file_logging}")


def _json_formatter(record: Dict[str, Any]) -> str:
    """Format log record as JSON."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "module": record["module"],
        "process": record["process"].id,
        "thread": record["thread"].id,
    }
    
    # Add extra fields from context
    if record.get("extra"):
        log_entry.update(record["extra"])
    
    # Add exception info if present
    if record.get("exception"):
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback
        }
    
    return json.dumps(log_entry, ensure_ascii=False)


class ContextualLogger:
    """Logger with additional context for request tracking."""
    
    def __init__(self, context: Dict[str, Any] = None):
        self.context = context or {}
        self.logger = logger.bind(**self.context)
    
    def add_context(self, **kwargs):
        """Add context to logger."""
        self.context.update(kwargs)
        self.logger = logger.bind(**self.context)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.bind(**kwargs).info(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.bind(**kwargs).error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.bind(**kwargs).warning(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.bind(**kwargs).debug(message)
    
    def exception(self, message: str, **kwargs):
        """Log exception with context."""
        self.logger.bind(**kwargs).exception(message)


def get_logger(name: str, context: Dict[str, Any] = None) -> ContextualLogger:
    """Get a contextual logger instance."""
    return ContextualLogger(context or {"module": name})