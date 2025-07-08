# Provider Configuration Guide

This document describes how to configure different providers for the MMCT Agent framework.

## Overview

The MMCT Agent framework supports multiple providers for different AI services:

- **LLM Providers**: Azure OpenAI, OpenAI
- **Search Providers**: Azure AI Search, Elasticsearch
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
from mmct.client_manager import ClientManager

config = MMCTConfig()
config.llm.provider = "azure"
config.llm.endpoint = "https://your-resource.openai.azure.com/"

client_manager = ClientManager(config)
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

### Basic Usage

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
from mmct.client_manager import ClientManager

# Create custom configuration
config = MMCTConfig()
config.llm.provider = "openai"
config.llm.api_key = "your-openai-key"

# Initialize client manager with custom config
client_manager = ClientManager(config)
```

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

### Code Changes

```python
# Legacy
from mmct.llm_client import LLMClient
client = LLMClient(service_provider="azure")

# New
from mmct.client_manager import get_client_manager
client_manager = get_client_manager()
llm_provider = client_manager.get_llm_provider()
```

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