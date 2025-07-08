from typing import Dict, Optional


class MMCTException(Exception):
    """Base exception for MMCT framework."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ProviderException(MMCTException):
    """Raised when external provider fails."""
    pass


class ConfigurationException(MMCTException):
    """Raised when configuration is invalid."""
    pass


class ValidationException(MMCTException):
    """Raised when input validation fails."""
    pass


class AuthenticationException(MMCTException):
    """Raised when authentication fails."""
    pass


class ResourceNotFoundException(MMCTException):
    """Raised when requested resource is not found."""
    pass