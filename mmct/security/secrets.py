import os
from typing import Optional
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from loguru import logger
from ..exceptions import AuthenticationException, ConfigurationException


class SecretsManager:
    """Manages secrets from Azure Key Vault or environment variables."""
    
    def __init__(self, vault_url: Optional[str] = None, use_managed_identity: bool = True):
        self.use_keyvault = vault_url is not None
        self.vault_url = vault_url
        self.client = None
        
        if self.use_keyvault:
            try:
                credential = DefaultAzureCredential() if use_managed_identity else None
                if credential is None:
                    raise ConfigurationException("Managed identity is disabled but no alternative credential provided")
                
                self.client = SecretClient(vault_url=vault_url, credential=credential)
                logger.info(f"Initialized Azure Key Vault client for {vault_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Key Vault client: {e}")
                raise AuthenticationException(f"Key Vault initialization failed: {e}")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret from Key Vault or environment variables.
        
        Args:
            key: Secret name/key
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        if self.use_keyvault and self.client:
            try:
                secret = self.client.get_secret(key)
                logger.debug(f"Retrieved secret '{key}' from Key Vault")
                return secret.value
            except Exception as e:
                logger.warning(f"Failed to retrieve secret '{key}' from Key Vault: {e}")
                
        # Fallback to environment variable
        value = os.getenv(key, default)
        if value:
            logger.debug(f"Retrieved secret '{key}' from environment variables")
        else:
            logger.warning(f"Secret '{key}' not found in Key Vault or environment variables")
        
        return value
    
    def set_secret(self, key: str, value: str) -> bool:
        """
        Set secret in Key Vault.
        
        Args:
            key: Secret name
            value: Secret value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_keyvault or not self.client:
            logger.warning("Key Vault not configured, cannot set secret")
            return False
            
        try:
            self.client.set_secret(key, value)
            logger.info(f"Successfully set secret '{key}' in Key Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret '{key}' in Key Vault: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        """
        Delete secret from Key Vault.
        
        Args:
            key: Secret name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_keyvault or not self.client:
            logger.warning("Key Vault not configured, cannot delete secret")
            return False
            
        try:
            self.client.begin_delete_secret(key)
            logger.info(f"Successfully deleted secret '{key}' from Key Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{key}' from Key Vault: {e}")
            return False