# Provider Configuration Guide

This document describes how to configure different providers for the MMCT Agent framework.

## Overview

The MMCT Agent framework supports multiple providers for different AI services:

- **LLM Providers**: Azure OpenAI, OpenAI
- **Embedding Providers**: Azure OpenAI, OpenAI (now separate from LLM providers)
- **Search Providers**: Azure AI Search
- **Vision Providers**: Azure Computer Vision, OpenAI Vision
- **Transcription Providers**: Azure Speech Service, OpenAI Whisper

## Configuration Methods

### 1. Environment Variables

The primary method for configuring providers is through environment variables. Copy one of the environment template files and update with your values:

```bash
# Development
cp config/environments/development.env .env

# Production
cp config/environments/production.env .env
```

### 2. Configuration Files

Providers can also be configured through YAML files:

```yaml
# config/providers.yaml
providers:
  llm:
    azure:
      endpoint: "https://your-resource.openai.azure.com/"
      deployment_name: "gpt-4o"
      # ... other settings
```

### 3. Programmatic Configuration

```python
from mmct.config.settings import MMCTConfig

config = MMCTConfig()
config.llm.provider = "azure"
config.llm.endpoint = "https://your-resource.openai.azure.com/"
```

## Provider-Specific Configuration

### LLM Providers

#### Azure OpenAI

```yaml
llm:
  provider: azure
  endpoint: "https://your-resource.openai.azure.com/"
  deployment_name: "gpt-4o"
  api_version: "2024-08-01-preview"
  model_name: "gpt-4o"
  use_managed_identity: true
  # api_key: "your-api-key"  # Only needed if use_managed_identity is false
```

**Environment Variables:**
- `LLM_PROVIDER=azure`
- `LLM_ENDPOINT=https://your-resource.openai.azure.com/`
- `LLM_DEPLOYMENT_NAME=gpt-4o`
- `LLM_API_VERSION=2024-08-01-preview`
- `LLM_MODEL_NAME=gpt-4o`
- `LLM_USE_MANAGED_IDENTITY=true`
- `LLM_API_KEY=your-api-key` (optional)

#### OpenAI

```yaml
llm:
  provider: openai
  api_key: "your-api-key"
  model_name: "gpt-4o"
  embedding_model: "text-embedding-3-small"
```

**Environment Variables:**
- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=your-api-key`
- `OPENAI_MODEL=gpt-4o`

### Embedding Providers

#### Azure OpenAI Embeddings

```yaml
embedding:
  provider: azure
  endpoint: "https://your-resource.openai.azure.com/"
  deployment_name: "text-embedding-3-small"
  api_version: "2024-08-01-preview"
  use_managed_identity: true
  # api_key: "your-api-key"  # Only needed if use_managed_identity is false
```

**Environment Variables:**
- `EMBEDDING_PROVIDER=azure`
- `EMBEDDING_SERVICE_ENDPOINT=https://your-resource.openai.azure.com/`
- `EMBEDDING_SERVICE_DEPLOYMENT_NAME=text-embedding-3-small`
- `EMBEDDING_SERVICE_API_VERSION=2024-08-01-preview`
- `EMBEDDING_USE_MANAGED_IDENTITY=true`
- `EMBEDDING_SERVICE_API_KEY=your-api-key` (optional)

#### OpenAI Embeddings

```yaml
embedding:
  provider: openai
  api_key: "your-api-key"
  embedding_model: "text-embedding-3-small"
```

**Environment Variables:**
- `EMBEDDING_PROVIDER=openai`
- `OPENAI_API_KEY=your-api-key`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

### Search Providers

#### Azure AI Search

```yaml
search:
  provider: azure_ai_search
  endpoint: "https://your-search.search.windows.net"
  index_name: "default"
  use_managed_identity: true
  # api_key: "your-search-key"  # Only needed if use_managed_identity is false
```

**Environment Variables:**
- `SEARCH_PROVIDER=azure_ai_search`
- `SEARCH_ENDPOINT=https://your-search.search.windows.net`
- `SEARCH_INDEX_NAME=default`
- `SEARCH_USE_MANAGED_IDENTITY=true`
- `SEARCH_API_KEY=your-search-key` (optional)

#### Elasticsearch

```yaml
search:
  provider: elasticsearch
  endpoint: "https://your-elasticsearch.com"
  username: "your-username"
  password: "your-password"
  index_name: "default"
```

**Environment Variables:**
- `SEARCH_PROVIDER=elasticsearch`
- `ELASTICSEARCH_ENDPOINT=https://your-elasticsearch.com`
- `ELASTICSEARCH_USERNAME=your-username`
- `ELASTICSEARCH_PASSWORD=your-password`
- `ELASTICSEARCH_INDEX_NAME=default`

### Vision Providers

#### Azure Computer Vision

```yaml
vision:
  provider: azure
  endpoint: "https://your-resource.openai.azure.com/"
  deployment_name: "gpt-4o"
  api_version: "2024-08-01-preview"
  use_managed_identity: true
```

**Environment Variables:**
- `VISION_PROVIDER=azure`
- `LLM_ENDPOINT=https://your-resource.openai.azure.com/`
- `LLM_VISION_DEPLOYMENT_NAME=gpt-4o`
- `LLM_VISION_API_VERSION=2024-08-01-preview`
- `LLM_USE_MANAGED_IDENTITY=true`

#### OpenAI Vision

```yaml
vision:
  provider: openai
  api_key: "your-api-key"
  model: "gpt-4o"
```

**Environment Variables:**
- `VISION_PROVIDER=openai`
- `OPENAI_API_KEY=your-api-key`
- `OPENAI_VISION_MODEL=gpt-4o`

### Transcription Providers

#### Azure Speech Service

```yaml
transcription:
  provider: azure
  endpoint: "https://your-speech.cognitiveservices.azure.com/"
  deployment_name: "whisper"
  api_version: "2024-08-01-preview"
  use_managed_identity: true
```

**Environment Variables:**
- `TRANSCRIPTION_PROVIDER=azure`
- `SPEECH_SERVICE_ENDPOINT=https://your-speech.cognitiveservices.azure.com/`
- `SPEECH_SERVICE_DEPLOYMENT_NAME=whisper`
- `SPEECH_SERVICE_API_VERSION=2024-08-01-preview`
- `SPEECH_USE_MANAGED_IDENTITY=true`

#### OpenAI Whisper

```yaml
transcription:
  provider: openai
  api_key: "your-api-key"
  model: "whisper-1"
```

**Environment Variables:**
- `TRANSCRIPTION_PROVIDER=openai`
- `OPENAI_API_KEY=your-api-key`
- `OPENAI_WHISPER_MODEL=whisper-1`

## Authentication

### Managed Identity (Recommended for Azure)

When using Azure services, managed identity is the recommended authentication method:

```yaml
use_managed_identity: true
```

This requires:
1. The application to be running in Azure (App Service, Container Apps, etc.)
2. A managed identity assigned to the service
3. Appropriate permissions granted to the managed identity

### API Keys

For development or non-Azure deployments, API keys can be used:

```yaml
use_managed_identity: false
api_key: "your-api-key"
```

### Azure Key Vault Integration

For production deployments, secrets can be stored in Azure Key Vault:

```yaml
security:
  keyvault_url: "https://your-keyvault.vault.azure.net/"
  enable_secrets_manager: true
```

## Usage Examples

### New Provider Pattern (Recommended)

```python
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig

# Initialize configuration
config = MMCTConfig()

# Create providers
llm_provider = provider_factory.create_llm_provider(
    config.llm.provider, 
    config.llm.model_dump()
)

embedding_provider = provider_factory.create_embedding_provider(
    config.embedding.provider,
    config.embedding.model_dump()
)

search_provider = provider_factory.create_search_provider(
    config.search.provider,
    config.search.model_dump()
)

# Use the providers
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = await llm_provider.chat_completion(messages)

# Generate embeddings
embedding = await embedding_provider.embedding("Hello world")
batch_embeddings = await embedding_provider.batch_embedding(["Text 1", "Text 2"])

# Search documents
results = await search_provider.search("artificial intelligence", "my-index")
```

### Legacy Client Manager (Deprecated)

```python
from mmct.client_manager import get_client_manager

# Get the global client manager
client_manager = get_client_manager()

# Use LLM provider
response = await client_manager.chat_completion([
    {"role": "user", "content": "Hello, how are you?"}
])

# Use search provider
results = await client_manager.search_documents("artificial intelligence")

# Use vision provider
with open("image.jpg", "rb") as f:
    image_data = f.read()
analysis = await client_manager.analyze_image(image_data)
```

### Custom Configuration

```python
from mmct.config.settings import MMCTConfig

# Create custom configuration
config = MMCTConfig()
config.llm.provider = "openai"
config.llm.api_key = "your-openai-key"
```

## Extending and Customizing Providers

### Creating Custom Providers

You can create custom providers by extending the base classes:

```python
from mmct.providers.base import LLMProvider, EmbeddingProvider
from mmct.providers.factory import provider_factory

class CustomLLMProvider(LLMProvider):
    def __init__(self, config):
        self.config = config
        # Initialize your custom provider
    
    async def chat_completion(self, messages, **kwargs):
        # Implement your custom logic
        return {
            "content": "Response from custom provider",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "custom-model",
            "finish_reason": "stop"
        }

# Register your custom provider
provider_factory.register_llm_provider("custom", CustomLLMProvider)

# Use it
custom_provider = provider_factory.create_llm_provider("custom", config)
```

### Extending Existing Providers

You can extend existing providers to add custom functionality:

```python
from mmct.providers.azure_providers import AzureLLMProvider

class EnhancedAzureLLMProvider(AzureLLMProvider):
    def __init__(self, config):
        super().__init__(config)
        self.request_counter = 0
    
    async def chat_completion(self, messages, **kwargs):
        self.request_counter += 1
        
        # Add custom pre-processing
        custom_system = {"role": "system", "content": "Enhanced AI assistant"}
        messages = [custom_system] + messages
        
        # Call parent method
        response = await super().chat_completion(messages, **kwargs)
        
        # Add custom metadata
        response["request_id"] = self.request_counter
        
        return response

# Register the enhanced provider
provider_factory.register_llm_provider("enhanced_azure", EnhancedAzureLLMProvider)
```

For complete examples, see:
- `examples/custom_provider_example.py` - Creating custom providers from scratch
- `examples/extend_provider_example.py` - Extending existing providers

## Environment Templates

The framework provides environment templates for different deployment scenarios:

- `config/environments/development.env`: Development configuration
- `config/environments/production.env`: Production configuration

Copy the appropriate template to `.env` and update with your specific values.

## Migration from Legacy Configuration

If you're migrating from the legacy `LLMClient` configuration, here's the mapping:

### Legacy Environment Variables

| Legacy | New | Notes |
|--------|-----|-------|
| `AZURE_OPENAI_ENDPOINT` | `LLM_ENDPOINT` | |
| `AZURE_OPENAI_DEPLOYMENT` | `LLM_DEPLOYMENT_NAME` | |
| `AZURE_OPENAI_API_KEY` | `LLM_API_KEY` | |
| `AZURE_AI_SEARCH_ENDPOINT` | `SEARCH_ENDPOINT` | |
| `AZURE_AI_SEARCH_KEY` | `SEARCH_API_KEY` | |
| `MANAGED_IDENTITY` | `LLM_USE_MANAGED_IDENTITY` | Now per-service |
| `EMBEDDING_SERVICE_ENDPOINT` | `EMBEDDING_SERVICE_ENDPOINT` | Now separate provider |
| `EMBEDDING_SERVICE_DEPLOYMENT_NAME` | `EMBEDDING_SERVICE_DEPLOYMENT_NAME` | Now separate provider |

### Code Changes

```python
# Legacy (deprecated)
from mmct.llm_client import LLMClient
client = LLMClient(service_provider="azure")

# New Provider Pattern (recommended)
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig

config = MMCTConfig()
llm_provider = provider_factory.create_llm_provider(
    config.llm.provider,
    config.llm.model_dump()
)

# For embeddings (now separate)
embedding_provider = provider_factory.create_embedding_provider(
    config.embedding.provider,
    config.embedding.model_dump()
)

# Legacy -> New method mapping
# client.get_client().chat.completions.create() -> llm_provider.chat_completion()
# client.get_client().embeddings.create() -> embedding_provider.embedding()
```

### Breaking Changes

1. **Separate Embedding Provider**: Embedding functionality is now separate from LLM providers
2. **New Response Format**: Provider methods return standardized response dictionaries
3. **Configuration Structure**: New configuration classes with validation
4. **Factory Pattern**: Providers are created through factory methods
5. **Deprecation Warnings**: `LLMClient` now shows deprecation warnings

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Ensure all required environment variables are set
2. **Authentication Failures**: Check managed identity permissions or API key validity
3. **Endpoint Errors**: Verify endpoint URLs and deployment names
4. **Model Not Found**: Ensure the specified model/deployment exists

### Debug Mode

Enable debug logging to troubleshoot configuration issues:

```bash
LOG_LEVEL=DEBUG
```

### Health Check

Use the health check endpoint to verify provider configuration:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/providers
```

## Security Best Practices

1. **Use Managed Identity**: Preferred for Azure deployments
2. **Store Secrets in Key Vault**: For production deployments
3. **Rotate API Keys**: Regularly rotate API keys
4. **Limit Permissions**: Grant minimum required permissions
5. **Use HTTPS**: Always use HTTPS endpoints
6. **Monitor Usage**: Monitor API usage and costs

## Support

For additional support or questions:

1. Check the [GitHub Issues](https://github.com/microsoft/MMCTAgent/issues)
2. Review the [Documentation](https://github.com/microsoft/MMCTAgent/docs)
3. Contact the development team