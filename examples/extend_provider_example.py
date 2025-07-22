"""
Example demonstrating how to extend existing providers to add custom functionality.

This example shows:
1. How to extend AzureLLMProvider to add custom features
2. How to extend OpenAIEmbeddingProvider to add custom behavior
3. How to use extended providers in the system
"""

from typing import Dict, Any, List
from mmct.providers.azure_providers import AzureLLMProvider
from mmct.providers.openai_providers import OpenAIEmbeddingProvider
from mmct.providers.factory import provider_factory
from mmct.exceptions import ProviderException
from loguru import logger


class ExtendedAzureLLMProvider(AzureLLMProvider):
    """Extended Azure LLM provider with custom features."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.custom_feature_enabled = config.get("custom_feature_enabled", False)
        self.request_counter = 0
        self.response_cache = {}
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Extended chat completion with caching and custom features."""
        self.request_counter += 1
        
        # Add custom logging
        logger.info(f"Request #{self.request_counter}: Custom Azure LLM provider called")
        
        # Simple caching mechanism (for demonstration)
        cache_key = str(hash(str(messages)))
        if cache_key in self.response_cache:
            logger.info("Returning cached response")
            return self.response_cache[cache_key]
        
        # Custom pre-processing
        if self.custom_feature_enabled:
            # Add custom system message
            custom_system = {
                "role": "system",
                "content": "You are an enhanced AI assistant with custom capabilities."
            }
            messages = [custom_system] + messages
        
        # Call parent method
        response = await super().chat_completion(messages, **kwargs)
        
        # Custom post-processing
        if self.custom_feature_enabled:
            # Add custom metadata
            response["custom_metadata"] = {
                "request_id": self.request_counter,
                "enhanced": True,
                "provider": "extended_azure"
            }
        
        # Cache the response
        self.response_cache[cache_key] = response
        
        return response


class ExtendedOpenAIEmbeddingProvider(OpenAIEmbeddingProvider):
    """Extended OpenAI embedding provider with custom normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.normalize_embeddings = config.get("normalize_embeddings", False)
        self.embedding_cache = {}
    
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Extended embedding with normalization and caching."""
        
        # Check cache first
        if text in self.embedding_cache:
            logger.info("Returning cached embedding")
            return self.embedding_cache[text]
        
        # Custom text preprocessing
        processed_text = self._preprocess_text(text)
        
        # Get embedding from parent
        embedding = await super().embedding(processed_text, **kwargs)
        
        # Custom post-processing
        if self.normalize_embeddings:
            embedding = self._normalize_vector(embedding)
        
        # Cache the result
        self.embedding_cache[text] = embedding
        
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """Custom text preprocessing."""
        # Example: remove extra whitespace, convert to lowercase
        return text.strip().lower()
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector to unit length."""
        import math
        magnitude = math.sqrt(sum(x**2 for x in vector))
        if magnitude > 0:
            return [x / magnitude for x in vector]
        return vector
    
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Extended batch embedding with custom processing."""
        logger.info(f"Processing batch of {len(texts)} texts")
        
        # Process each text individually to leverage caching
        embeddings = []
        for text in texts:
            embedding = await self.embedding(text, **kwargs)
            embeddings.append(embedding)
        
        return embeddings


def register_extended_providers():
    """Register extended providers with the factory."""
    
    # Register extended providers with new names
    provider_factory.register_llm_provider(
        name="extended_azure",
        provider_class=ExtendedAzureLLMProvider
    )
    
    provider_factory.register_embedding_provider(
        name="extended_openai",
        provider_class=ExtendedOpenAIEmbeddingProvider
    )
    
    print("Extended providers registered successfully!")


async def example_usage():
    """Example of how to use extended providers."""
    
    # Register extended providers
    register_extended_providers()
    
    # Configuration for extended providers
    azure_config = {
        "endpoint": "https://your-azure-endpoint.openai.azure.com/",
        "deployment_name": "your-deployment",
        "api_version": "2024-08-01-preview",
        "api_key": "your-api-key",
        "use_managed_identity": False,
        "custom_feature_enabled": True,  # Custom parameter
        "timeout": 200
    }
    
    openai_config = {
        "api_key": "your-openai-api-key",
        "embedding_model": "text-embedding-3-small",
        "normalize_embeddings": True,  # Custom parameter
        "timeout": 200
    }
    
    try:
        # Create extended providers
        extended_llm = provider_factory.create_llm_provider("extended_azure", azure_config)
        extended_embedding = provider_factory.create_embedding_provider("extended_openai", openai_config)
        
        # Test extended LLM provider
        messages = [{"role": "user", "content": "Hello, extended provider!"}]
        llm_response = await extended_llm.chat_completion(messages)
        print(f"Extended LLM Response: {llm_response}")
        
        # Test extended embedding provider
        test_texts = ["Hello world", "Test embedding", "Hello world"]  # Note: duplicate for cache test
        
        # Single embedding
        embedding1 = await extended_embedding.embedding(test_texts[0])
        print(f"Single embedding length: {len(embedding1)}")
        
        # Batch embedding (will use caching)
        batch_embeddings = await extended_embedding.batch_embedding(test_texts)
        print(f"Batch embeddings count: {len(batch_embeddings)}")
        
        # Test caching - should be faster
        embedding2 = await extended_embedding.embedding(test_texts[0])
        print(f"Cached embedding matches: {embedding1 == embedding2}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())