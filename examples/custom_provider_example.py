"""
Example demonstrating how to create custom providers and extend the provider system.

This example shows:
1. How to create a custom LLM provider
2. How to create a custom embedding provider
3. How to register custom providers with the factory
4. How to use custom providers in the system
"""

from typing import Dict, Any, List
from mmct.providers.base import LLMProvider, EmbeddingProvider
from mmct.providers.factory import provider_factory
from mmct.exceptions import ProviderException


class CustomLLMProvider(LLMProvider):
    """Example custom LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "custom-model")
        
        if not self.api_key:
            raise ProviderException("API key is required for CustomLLMProvider")
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using custom provider."""
        try:
            # Simulate custom LLM provider logic
            # Replace this with actual API calls to your custom provider
            
            # Example response format
            response = {
                "content": "This is a response from the custom LLM provider",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "model": self.model_name,
                "finish_reason": "stop"
            }
            
            return response
        except Exception as e:
            raise ProviderException(f"Custom LLM provider failed: {e}")


class CustomEmbeddingProvider(EmbeddingProvider):
    """Example custom embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "custom-embedding-model")
        
        if not self.api_key:
            raise ProviderException("API key is required for CustomEmbeddingProvider")
    
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using custom provider."""
        try:
            # Simulate custom embedding provider logic
            # Replace this with actual API calls to your custom provider
            
            # Example: return a dummy embedding vector
            return [0.1] * 768  # Typical embedding dimension
        except Exception as e:
            raise ProviderException(f"Custom embedding provider failed: {e}")
    
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts using custom provider."""
        try:
            # Simulate batch embedding
            return [[0.1] * 768 for _ in texts]
        except Exception as e:
            raise ProviderException(f"Custom batch embedding failed: {e}")


def register_custom_providers():
    """Register custom providers with the factory."""
    
    # Register the custom LLM provider
    provider_factory.register_llm_provider(
        name="custom_llm",
        provider_class=CustomLLMProvider
    )
    
    # Register the custom embedding provider
    provider_factory.register_embedding_provider(
        name="custom_embedding", 
        provider_class=CustomEmbeddingProvider
    )
    
    print("Custom providers registered successfully!")


async def example_usage():
    """Example of how to use custom providers."""
    
    # Register custom providers
    register_custom_providers()
    
    # Check supported providers
    supported = provider_factory.get_supported_providers()
    print(f"Supported providers: {supported}")
    
    # Create custom provider instances
    llm_config = {
        "api_key": "your-custom-api-key",
        "model_name": "custom-model-v1"
    }
    
    embedding_config = {
        "api_key": "your-custom-api-key",
        "model_name": "custom-embedding-v1"
    }
    
    # Create providers
    custom_llm = provider_factory.create_llm_provider("custom_llm", llm_config)
    custom_embedding = provider_factory.create_embedding_provider("custom_embedding", embedding_config)
    
    # Use the custom providers
    messages = [{"role": "user", "content": "Hello, custom provider!"}]
    
    try:
        # Test LLM provider
        llm_response = await custom_llm.chat_completion(messages)
        print(f"LLM Response: {llm_response}")
        
        # Test embedding provider
        embedding = await custom_embedding.embedding("Test text")
        print(f"Embedding length: {len(embedding)}")
        
        # Test batch embedding
        batch_embeddings = await custom_embedding.batch_embedding(["Text 1", "Text 2"])
        print(f"Batch embeddings count: {len(batch_embeddings)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())