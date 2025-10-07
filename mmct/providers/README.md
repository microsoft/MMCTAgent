# **Providers Module**

A flexible and extensible provider system for integrating multiple AI services including LLMs, embedding models, image generation, vision, and transcription services.

## 📁 Architecture

The providers module is organized into three main components:

```py
providers/
├── base/                  # Base provider classes (abstract interfaces)
├── azure_providers/       # Azure service implementations
├── custom_providers/      # Your custom provider implementation should be added here
└── factory.py             # provider availability
```

### Base Providers

The `base/` folder contains abstract base classes that define the interface for each provider type:

- **`llm_provider.py`** - Base class for Language Model providers
- **`embedding_provider.py`** - Base class for Embedding providers
- **`search_provider.py`** - Base class for Search Index provider
- **`vision_provider.py`** - Base class for Vision/Image Understanding providers
- **`transcription_provider.py`** - Base class for Audio Transcription providers

### Azure Providers

The `azure_providers/` module contains ready-to-use implementations of all base providers using Azure services. These serve as reference implementations and can be used directly in your projects.

By default, the provider usage is set for Azure Resources. Kindly go through the `.env` to fill out the required Azure Config.

### Custom Providers

The `custom_providers/` module is where you add your own provider implementations for any custom services.

---

## 🚀 Adding a Custom Provider

Follow these steps to add your own provider implementation (search provider example is given below):

### Step 1: Inherit from Base Provider

Choose the appropriate base provider class from the `base/` folder and perform relevant implementation. Below is an example for the **Custom Search Provider**

**Example structure of custom search provider:**

```python
# providers/custom_providers/search_provider.py
from mmct.providers.base import SearchProvider

class CustomSearchProvider(SearchProvider):
    """Custom class for search providers."""
    
    async def search(self, query: str, index_name: str, **kwargs) -> List[Dict]:
        """Search for documents."""
        pass
    
    async def index_document(self, document: Dict, index_name: str) -> bool:
        """Index a document."""
        pass
    
    async def delete_document(self, doc_id: str, index_name: str) -> bool:
        """Delete a document."""
        pass
```
_Add the relevant implementation for your search provider._

### Step 2: Add to Custom Providers Module

Place your implementation file in the `custom_providers/` folder:

```
custom_providers/
├── __init__.py
└── search_provider.py
```

### Step 3: Update `__init__.py`

Export your provider class in `custom_providers/__init__.py`:

```python
from .search_provider import CustomSearchProvider
from .llm_provider import CustomLLMProvider

__all__ = [
    'CustomSearchProvider',
    'CustomLLMProvider',
]
```

### Step 4: Register the provider in the factory

Add your provider to the available providers list in `mmct/providers/factory.py`:

```python
from .azure_providers import (
    AzureSearchProvider
)
from .custom_providers import (
    CustomSearchProvider
)

class ProviderFactory:
    """Factory class for creating provider instances."""

    _search_providers: Dict[str, Type[SearchProvider]] = {
            'azure_ai_search': AzureSearchProvider,
            'custom_search': CustomSearchProvider   # <-----      
            # Add other search providers here
        }
```

### Step 5: Add Environment Variables

If your provider requires additional configuration, add the environment variables to your `.env` file and update the `mmct/config/settings.py`:

```env
# Custom Provider Configuration
CUSTOM_API_KEY=your_api_key_here
CUSTOM_ENDPOINT=https://api.example.com
CUSTOM_MODEL_NAME=model-name
```

Then update the relevant Configs available in the config/settings.py

---

## 🔧 Using Providers

There are three ways to use providers in your application:

### Method 1: Environment Variables (Recommended)

Set your preferred provider in the `.env` file:

```env
LLM_PROVIDER=azure
EMBEDDING_PROVIDER=openai
SEARCH_PROVIDER=custom_search
```

The application will automatically load the configured provider.

### Method 2: Configuration File

Update `mmct/config/settings.py` to set default providers:

Under the respective config (example `LLMConfig`, `SearchConfig` etc) you can directly assign the by default provider name. Example:

```python
class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    
    provider: str = Field(default="azure", env="LLM_PROVIDER")
```

_You can add the relevant environment variables that you have to provider to your custom implementation, you can add those variables directly in the respective config itself_

### Method 3: Direct Instantiation

Load providers directly in your code:

```python
# importing the search config and the provider_factory
from mmct.config.settings import SearchConfig
from mmct.providers.factory import provider_factory

# setting the provider directly available in the provider_factory
search_config = SearchConfig(provider="custom_search")

# instantiating the provider class according the config.
search_provider = provider_factory.create_search_provider(
    search_config.provider, search_config.model_dump()
)

# performing the search over the `query` and the generated `query-embedding`
search_results = await search_provider.search(query=query,embedding=embedding)
```

---

## 🎯 Best Practices

1. **Error Handling**: Always implement proper error handling in your provider methods
2. **Configuration Validation**: Validate required configuration parameters in `__init__`
3. **Documentation**: Add docstrings to your provider classes and methods
4. **Testing**: Create unit tests for your custom providers
5. **Environment Variables**: Use environment variables for sensitive information like API keys

---

## 🤝 Contributing

When contributing a new provider implementation:

1. Follow the existing code structure and naming conventions
2. Implement all required methods from the base class
3. Add comprehensive documentation
4. Include usage examples
5. Test thoroughly with different scenarios

---