import asyncio
import functools
from typing import TypeVar, Callable, Any, Optional, Type, Union
from loguru import logger
from ..exceptions import MMCTException, ProviderException

T = TypeVar('T')


def handle_exceptions(
    retries: int = 3,
    fallback: Any = None,
    exceptions: Union[Type[Exception], tuple] = Exception,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """
    Decorator to handle exceptions with retry logic and fallback.
    
    Args:
        retries: Number of retry attempts
        fallback: Fallback value to return if all retries fail
        exceptions: Exception types to catch and retry
        backoff_factor: Exponential backoff factor
        max_delay: Maximum delay between retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
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
        def sync_wrapper(*args, **kwargs) -> T:
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
    """
    Decorator to log exceptions.
    
    Args:
        log_level: Log level for exception logging
        include_traceback: Whether to include traceback in log
        custom_message: Custom message to include in log
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
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
        def sync_wrapper(*args, **kwargs) -> T:
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


def convert_exceptions(exception_map: dict):
    """
    Decorator to convert exceptions to MMCT exceptions.
    
    Args:
        exception_map: Dictionary mapping exception types to MMCT exception types
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                for source_exc, target_exc in exception_map.items():
                    if isinstance(e, source_exc):
                        raise target_exc(str(e), details={"original_exception": type(e).__name__})
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
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
    """Centralized error handling utilities."""
    
    @staticmethod
    def handle_provider_error(e: Exception, provider_name: str) -> ProviderException:
        """Convert provider-specific exceptions to ProviderException."""
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
    def log_and_raise(exception: Exception, context: str = ""):
        """Log exception and re-raise it."""
        message = f"{context}: {exception}" if context else str(exception)
        logger.error(message)
        raise exception