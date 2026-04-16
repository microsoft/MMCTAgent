"""Centralized exception handling and error management.

This module provides a unified exception hierarchy for the MMCT framework, 
along with decorators for automated retry logic, exception conversion, 
and logging.
"""

import asyncio
import functools
from typing import TypeVar, Callable, Any, Optional, Type, Union, Dict, List
from loguru import logger

class MMCTException(Exception):
    """Base exception for the MMCT framework.

    All custom exceptions within the framework should inherit from this class 
    to provide consistent error code and metadata tracking.

    Attributes:
        message (str): A descriptive error message.
        error_code (str, optional): A unique alphanumeric code for the error type.
        details (Dict, optional): Additional metadata or context about the error.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ProviderException(MMCTException):
    """Raised when an external provider (LLM, DB, Storage) fails."""
    pass


class ConfigurationException(MMCTException):
    """Raised when configuration values are missing or invalid."""
    pass


class ValidationException(MMCTException):
    """Raised when input parameter validation fails."""
    pass


class AuthenticationException(MMCTException):
    """Raised when authentication with an external service fails."""
    pass


class ResourceNotFoundException(MMCTException):
    """Raised when a requested resource (file, node, video) is not found."""
    pass


T = TypeVar('T')


def handle_exceptions(
    retries: int = 3,
    fallback: Any = None,
    exceptions: Union[Type[Exception], tuple] = Exception,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """Decorator to handle exceptions with retry logic and optional fallback.

    Supports both synchronous and asynchronous functions. Implements 
    exponential backoff between retry attempts.

    Args:
        retries: Total number of attempts (including the first call).
        fallback: Value to return if all retry attempts fail.
        exceptions: Exception type or tuple of types to catch and retry.
        backoff_factor: Multiplier for exponential backoff (delay = factor ** attempt).
        max_delay: Maximum delay in seconds between retries.

    Returns:
        Callable: The decorated function with retry logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < retries - 1:
                        delay = min(backoff_factor ** attempt, max_delay)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {retries} attempts failed: {e}")
            
            if fallback is not None:
                logger.info(f"Returning fallback value: {fallback}")
                return fallback
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < retries - 1:
                        delay = min(backoff_factor ** attempt, max_delay)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"All {retries} attempts failed: {e}")
            
            if fallback is not None:
                logger.info(f"Returning fallback value: {fallback}")
                return fallback
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_exceptions(
    log_level: str = "ERROR",
    include_traceback: bool = True,
    custom_message: Optional[str] = None
):
    """Decorator to log any exceptions raised by the decorated function.

    Args:
        log_level: The loguru level to use for logging.
        include_traceback: If True, includes a full stack trace in the log.
        custom_message: Optional prefix or override for the log message.

    Returns:
        Callable: The decorated function with logging logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                message = custom_message or f"Exception in {func.__name__}"
                if include_traceback:
                    logger.opt(exception=True).log(log_level, f"{message}: {e}")
                else:
                    logger.log(log_level, f"{message}: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = custom_message or f"Exception in {func.__name__}"
                if include_traceback:
                    logger.opt(exception=True).log(log_level, f"{message}: {e}")
                else:
                    logger.log(log_level, f"{message}: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def convert_exceptions(exception_map: Dict[Type[Exception], Type[MMCTException]]):
    """Decorator to convert low-level exceptions to MMCT framework exceptions.

    Args:
        exception_map: A dictionary mapping source exception types to target 
            MMCTException types.

    Returns:
        Callable: The decorated function with exception conversion.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                for source_exc, target_exc in exception_map.items():
                    if isinstance(e, source_exc):
                        raise target_exc(str(e), details={"original_exception": type(e).__name__})
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                for source_exc, target_exc in exception_map.items():
                    if isinstance(e, source_exc):
                        raise target_exc(str(e), details={"original_exception": type(e).__name__})
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ErrorHandler:
    """Centralized error handling and conversion utilities."""
    
    @staticmethod
    def handle_provider_error(e: Exception, provider_name: str) -> ProviderException:
        """Converts a provider-specific exception into a ProviderException.

        Args:
            e: The original exception from the provider client.
            provider_name: The name of the provider (e.g., 'Azure', 'Neo4j').

        Returns:
            ProviderException: A wrapped framework exception.
        """
        error_details = {
            "provider": provider_name,
            "original_exception": type(e).__name__,
            "message": str(e)
        }
        
        logger.error(f"Provider {provider_name} error: {e}")
        return ProviderException(
            f"Provider {provider_name} failed: {e}",
            error_code="PROVIDER_ERROR",
            details=error_details
        )
    
    @staticmethod
    def log_and_raise(exception: Exception, context: str = "") -> None:
        """Logs an exception with optional context and re-raises it.

        Args:
            exception: The exception object to log.
            context: Optional prefix for the log message.

        Raises:
            Exception: The same exception passed as an argument.
        """
        message = f"{context}: {exception}" if context else str(exception)
        logger.error(message)
        raise exception