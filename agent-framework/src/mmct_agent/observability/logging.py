"""Structured logging configuration using loguru."""

from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


# Context variable for trace ID (thread-safe and async-safe)
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

# Track if logging has been configured
_logging_configured: bool = False


@dataclass
class LogConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    format: str = "text"  # "json" or "text"
    include_timestamp: bool = True
    include_trace_id: bool = True
    output: str = "stderr"  # "stderr", "stdout", or file path
    colorize: bool = True  # Enable colored output for text format


def _get_trace_id() -> str:
    """Get current trace ID or default."""
    return _trace_id_var.get() or "-"


def _json_sink(message: "Record") -> None:
    """Custom sink that outputs JSON formatted logs."""
    record = message.record
    
    log_data: dict[str, object] = {
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
    }
    
    # Add timestamp
    log_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Add trace ID
    trace_id = _get_trace_id()
    if trace_id != "-":
        log_data["trace_id"] = trace_id
    
    # Add extra fields from record
    if record["extra"]:
        for key, value in record["extra"].items():
            if key not in ("trace_id",):
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
    
    # Add exception info if present
    if record["exception"]:
        log_data["exception"] = "".join(
            record["exception"].traceback.format() if record["exception"].traceback else []
        )
    
    sys.stderr.write(json.dumps(log_data, default=str) + "\n")


def _text_format(record: "Record") -> str:
    """Format log record as text with trace ID."""
    trace_id = _get_trace_id()
    
    # Base format with color support
    time_fmt = "<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> | " if True else ""
    trace_fmt = f"<cyan>[{trace_id}]</cyan> | " if trace_id != "-" else ""
    
    return (
        f"{time_fmt}"
        f"<level>{{level: <8}}</level> | "
        f"{trace_fmt}"
        f"<blue>{{name}}</blue> | "
        f"{{message}}\n"
        f"{{exception}}"
    )


def _text_format_no_timestamp(record: "Record") -> str:
    """Format log record as text without timestamp."""
    trace_id = _get_trace_id()
    trace_fmt = f"<cyan>[{trace_id}]</cyan> | " if trace_id != "-" else ""
    
    return (
        f"<level>{{level: <8}}</level> | "
        f"{trace_fmt}"
        f"<blue>{{name}}</blue> | "
        f"{{message}}\n"
        f"{{exception}}"
    )


def setup_logging(config: LogConfig | None = None) -> None:
    """Set up logging with the given configuration.
    
    Args:
        config: Logging configuration.
    """
    global _logging_configured
    config = config or LogConfig()
    
    # Remove all handlers (including default)
    logger.remove()
    
    # Determine output sink
    if config.output == "stderr":
        sink = sys.stderr
    elif config.output == "stdout":
        sink = sys.stdout
    else:
        sink = config.output  # File path
    
    # Filter function for mmct_agent logs
    def mmct_filter(record: dict) -> bool:
        name = record.get("extra", {}).get("name", "")
        return name.startswith("mmct_agent") or name == "mmct_agent"
    
    # Configure based on format
    if config.format == "json":
        # Use custom JSON sink
        logger.add(
            _json_sink,
            level=config.level.upper(),
            filter=mmct_filter,
            colorize=False,
        )
    else:
        # Use text format
        format_func = _text_format if config.include_timestamp else _text_format_no_timestamp
        
        logger.add(
            sink,
            format=format_func,
            level=config.level.upper(),
            filter=mmct_filter,
            colorize=config.colorize and config.output in ("stderr", "stdout"),
        )
    
    _logging_configured = True


def get_logger(name: str):
    """Get a logger for the given name.
    
    Args:
        name: Logger name (will be prefixed with 'mmct_agent.').
        
    Returns:
        Logger instance bound to the name.
    """
    global _logging_configured
    
    if not name.startswith("mmct_agent"):
        name = f"mmct_agent.{name}"
    
    # If logging hasn't been configured yet, set up a default configuration
    # that suppresses logs until explicit setup
    if not _logging_configured:
        logger.remove()
        _logging_configured = True
        
    return logger.bind(name=name)


def set_trace_id(trace_id: str) -> None:
    """Set the current trace ID for logging.
    
    Args:
        trace_id: Trace ID to use.
    """
    _trace_id_var.set(trace_id)


def clear_trace_id() -> None:
    """Clear the current trace ID."""
    _trace_id_var.set(None)
