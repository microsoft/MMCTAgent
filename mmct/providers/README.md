# **Providers Module**

A flexible and extensible provider system for integrating multiple services including LLMs, embedding models, image embedding, storage, and transcription services.

## 📁 Architecture

The providers module is organized into three main components:

```py
providers/
├── base/                  # Base provider classes (abstract interfaces)
├── azure_providers/       # Azure service implementations
├── custom_providers/      # local provider implementation
```

### Base Providers

The `base/` folder contains abstract base classes that define the interface for each provider type:

- **`llm_provider.py`** - Base class for Language Model providers
- **`embedding_provider.py`** - Base class for Embedding providers
- **`search_provider.py`** - Base class for Search Index provider
- **`transcription_provider.py`** - Base class for Audio Transcription providers
- **`image_embedding_provider`** - Base class for image embedding generation providers
- **`storage_provider`** - Base class for the storage providers.

### Azure Providers

The `azure_providers/` module contains ready-to-use implementations of all base providers using Azure services. These serve as reference implementations and can be used directly in your projects.
---

## 🔌 Implementing Custom Provider
Here is the implementation plan for custom LLM Provider

MMCTAgent's provider system is fully extensible. You can implement custom LLM providers in your own codebase to support any LLM vendor (e.g., Anthropic, Cohere, Hugging Face, etc.).

### Required Methods

All LLM providers must implement two abstract methods from `BaseLLMProvider`:

1. **`async chat_completion(messages: List[Dict], **kwargs) -> Dict[str, Any]`**
   - Generates chat completions using the LLM
   - Must return a dict with: `content`, `usage`, `model`, `finish_reason`

2. **`get_autogen_client(**kwargs)`**
   - Returns an autogen-compatible client for the LLM
   - Required for integration with autogen-based agents (ImageAgent, VideoAgent, IngestionPipeline)

### Implementation Steps

#### Step 1: Create Your Provider Class in Your Codebase

```python
# Example: your_project/providers/anthropic_provider.py
from mmct.providers.base import BaseLLMProvider
from typing import Dict, Any, List, Optional
import anthropic

class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation for Claude models."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-3-5-sonnet-20241022",
        timeout: Optional[int] = 600,
        max_retries: Optional[int] = 2,
    ):
        if not api_key:
            raise ValueError("Anthropic API key is required!")

        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Anthropic Claude API."""
        # See examples/image_agent.ipynb for complete implementation
        pass

    def get_autogen_client(self, **kwargs):
        """Get autogen-compatible client for Anthropic."""
        try:
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient

            temperature = kwargs.get("temperature", 1.0)
            max_tokens = kwargs.get("max_tokens", 4096)

            return AnthropicChatCompletionClient(
                model=self.model_name,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            raise Exception(
                "autogen_ext.models.anthropic is not available. "
                "Install with: pip install 'autogen-ext[anthropic]'"
            )

    async def close(self):
        """Close the Anthropic client and cleanup resources."""
        if self.client:
            await self.client.close()
```

**📓 Complete Working Example**: See [`examples/image_agent.ipynb`](../../examples/image_agent.ipynb) for the full implementation with message conversion, error handling, and all required details.

#### Step 2: Use Your Provider with Agents

Simply instantiate your provider and pass it to ImageAgent, VideoAgent, or IngestionPipeline:

```python
from mmct.config.providers import ImageAgentProviderConfig, VideoAgentProviderConfig, IngestionProviderConfig
from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mmct.video_pipeline import VideoAgent, IngestionPipeline
from your_project.providers.anthropic_provider import AnthropicLLMProvider

# Create your custom LLM provider instance
custom_llm = AnthropicLLMProvider(
    api_key="your-anthropic-api-key",
    model_name="claude-3-5-sonnet-20241022",
)

# Use with ImageAgent
image_provider = ImageAgentProviderConfig(llm_provider=custom_llm)
image_agent = ImageAgent(
    query="What objects are visible in this image?",
    image_path="path/to/image.jpg",
    tools=[ImageQnaTools.vit, ImageQnaTools.object_detection],
    provider=image_provider
)
response = await image_agent()

# Use with VideoAgent
video_provider = VideoAgentProviderConfig(
    llm_provider=custom_llm,
    # You can also pass other providers like search_provider, embedding_provider, etc.
)
video_agent = VideoAgent(
    query="What happens in the video?",
    index_name="your-index",
    provider=video_provider
)
response = await video_agent()

# Use with IngestionPipeline
ingestion_provider = IngestionProviderConfig(
    llm_provider=custom_llm,
    # You can also pass other providers like search_provider, embedding_provider, storage_provider, etc.
)
video_path="path/to/video.mp4"
video_id = "some-video-id" # or calculate hash

ingestion = IngestionPipeline(
    video_path=video_path,
    video_id=video_id,
    provider=ingestion_provider
)
await ingestion.run()
```

### Key Implementation Notes

1. **Message Format Conversion**: Different LLM APIs use different message formats:
   - OpenAI: System messages are part of the messages array
   - Anthropic: System messages are a separate parameter
   - Your provider should handle these conversions internally

2. **Error Handling**: Implement robust error handling, especially in `get_autogen_client()` for cases where autogen-ext doesn't support your vendor yet

3. **Response Format**: Ensure `chat_completion()` returns a consistent format:
   ```python
   {
       "content": str,           # The response text
       "usage": {                # Token usage stats
           "prompt_tokens": int,
           "completion_tokens": int,
           "total_tokens": int
       },
       "model": str,             # Model used
       "finish_reason": str      # Why generation stopped
   }
   ```

4. **Async Support**: Use async clients (e.g., `AsyncAnthropic`) for better performance

---

## 🚀 Adding Other Custom Providers

Follow these steps to add your own provider implementation (search provider example is given below):

### Step 1: Inherit from Base Provider

Choose the appropriate base provider class from the `base/` folder and perform relevant implementation. Below is an example for the **Custom Search Provider**

**Example structure of custom search provider:**

```python
from mmct.providers.base import AISearchObjectCollectionProvider

class CustomSearchProvider(AISearchObjectCollectionProvider):
    """Custom class for search providers."""

    def get_index_schema(self) -> Any:
        """Creates provider-specific schema based on ObjectCollectionDocument type."""
        pass

    def parse_response(self, vector_db_document: Any) -> ObjectCollectionDocument:
        """Parses the retrieved vector DB document into ObjectCollectionDocument object."""
        pass

    async def search(self, query: str, **kwargs) -> List[Tuple[ObjectCollectionDocument, float]]:
        """Search for documents."""
        pass

    async def index_document(self, document: Dict) -> bool:
        """Index a document."""
        pass

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        pass

    async def create_index(self) -> bool:
        """Create a index with the given schema."""
        pass

    async def index_exists(self) -> bool:
        """Check if an index exists."""
        pass

    async def delete_index(self) -> bool:
        """Delete a index."""
        pass

    async def upload_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """Upload multiple documents to the index."""
        pass

    async def check_is_document_exist(self, hash_id: str) -> bool:
        """Check if a document with the given hash_id exists in the index."""
        pass

    async def close(self):
        """Close the client and cleanup resources. Optional to implement."""
        pass
```

_Add the relevant implementation for your search provider._

### Step 2: Use Your Provider

Simply instantiate and use your provider directly in your code. Follow the same pattern as shown in the LLM provider example above - create an instance and pass it to the appropriate config.

---

## 🔧 Using Custom Providers

Custom providers are used via direct instantiation. See the usage examples in the sections above:
- For LLM providers: See [Implementing Custom LLM Providers](#-implementing-custom-llm-providers)
- For other providers: Follow the same pattern - instantiate your provider and pass it to the appropriate agent config

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
