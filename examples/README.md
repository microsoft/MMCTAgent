# MMCT Agent Examples

This directory contains example notebooks demonstrating how to use the MMCT Agent framework components.

## 📓 Available Examples

### 1. **image_agent.ipynb**
Demonstrates how to use the `ImageAgent` for image analysis tasks.

**Features:**
- Image analysis using multiple tools (OCR, Object Detection, VIT, RECOG)
- Optional critic agent for improved accuracy
- Configurable tool selection

**Usage:**
```python
from mmct.image_pipeline import ImageAgent, ImageQnaTools

agent = ImageAgent(
    query="describe the image",
    image_path="path/to/image.jpg",
    tools=[ImageQnaTools.OCR, ImageQnaTools.VIT],
    use_critic_agent=True
)
response = await agent()
```

### 2. **video_agent.ipynb**
Shows how to use the `VideoAgent` for video question answering.

**Features:**
- Video retrieval from Azure AI Search index
- Multi-modal analysis using MMCT framework
- Support for Computer Vision integration
- Configurable critic agent

**Usage:**
```python
from mmct.video_pipeline import VideoAgent

agent = VideoAgent(
    query="what is discussed in the video?",
    index_name="video-index",
    top_n=2,
    use_computer_vision_tool=False,
    use_critic_agent=True
)
response = await agent()
```

### 3. **ingestion_pipeline.ipynb**
Demonstrates the video ingestion pipeline for processing videos.

**Features:**
- Video transcription using Whisper or Azure Speech-to-Text
- Frame extraction and chapter generation
- Azure AI Search indexing
- Optional Computer Vision integration

**Usage:**
```python
from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices

pipeline = IngestionPipeline(
    video_path="path/to/video.mp4",
    index_name="video-index",
    transcription_service=TranscriptionServices.WHISPER,
    language=Languages.ENGLISH_INDIA,
    use_computer_vision_tool=False
)
await pipeline()
```

### 4. **custom_provider_example.py**
Demonstrates how to create custom providers and extend the provider system.

**Features:**
- Creating custom LLM and embedding providers
- Registering custom providers with the factory
- Using custom providers in the system
- Provider configuration and error handling

**Usage:**
```python
from examples.custom_provider_example import register_custom_providers

# Register your custom providers
register_custom_providers()

# Use them through the factory
from mmct.providers.factory import provider_factory
custom_llm = provider_factory.create_llm_provider("custom_llm", config)
```

### 5. **extend_provider_example.py**
Shows how to extend existing providers to add custom functionality.

**Features:**
- Extending AzureLLMProvider with custom features
- Extending OpenAIEmbeddingProvider with normalization
- Adding caching and custom processing
- Preserving base provider functionality

**Usage:**
```python
from examples.extend_provider_example import register_extended_providers

# Register extended providers
register_extended_providers()

# Use extended providers
extended_llm = provider_factory.create_llm_provider("extended_azure", config)
```

## 🛠️ Setup Requirements

Before running these examples:

1. **Environment Setup**: Create a `.env` file in the root directory with required Azure credentials
2. **Install Dependencies**: Run `pip install -r requirements.txt`
3. **Azure Services**: Ensure you have access to required Azure services (OpenAI, Storage, Search, etc.)

## 📋 Notes

- All examples use `nest_asyncio` to support async operations in Jupyter notebooks
- Update file paths in the examples to match your local setup
- The examples assume you have properly configured Azure services and credentials