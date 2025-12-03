# **MMCT - Video Pipeline**

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.18358-b31b1b.svg)](https://arxiv.org/abs/2405.18358)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2405.18358">
    <img src="/docs/multimedia/mmct-video-pipeline.png" alt="Video Pipeline - Main Architecture" width="95%" />
  </a>
</p>

## **Overview**

MMCTAgent's Video Pipeline is a state-of-the-art multi-modal AI framework that brings human-like critical thinking to video understanding tasks. It consists of two main components:

1. **Video Ingestion** - Comprehensive video processing and preparation
2. **Video Agent** - Multi-modal critical thinking for video question answering

The pipeline processes videos through ingestion, then applies the **Multi-Modal Critical Thinking (MMCT)** framework for sophisticated video understanding and question answering. The **Video Agent** leverages structured reasoning with planner and critic components to deliver superior performance in complex video analysis tasks.

## **1. Video Ingestion**

The **IngestionPipeline** performs comprehensive processing of video file to extract transcript, frames, chapters, ai search index creation for downstream applications like `VideoAgent`. It includes the following steps:

1. **Audio Extraction** – Extracts the audio from the input video.
2. **Transcription** – Converts spoken content to text using the selected transcription service and language setting.

    > Transcription Configuration

    You can configure the transcription backend using the `TranslationServices` enum:

    - `TranslationServices.WHISPER` – Uses OpenAI Whisper.
    - `TranslationServices.AZURE_STT` – Uses Azure Speech-to-Text.

    Specify the language of the video's audio using the `Languages` enum. For example:

    - `Languages.ENGLISH_INDIA` – English (India)
    - `Languages.HINDI` – Hindi

    The `Languages` enum includes support for additional languages. Refer to the `Languages` enum definition to explore all available options.

3. **Frame Extraction** – Captures representative frames at 1 FPS intervals to support visual summarization and downstream VideoAgent.
4. **Chapter Generation** – Aligns transcript segments with visual frames to form meaningful video chapters.
5. **Azure Search Indexing** – Saves chapters and metadata to an Azure AI Search index to support retrieval.
6. **Summary File Generation** – Outputs `summary_n_transcript.json` containing the full transcript and a visual summary.
7. **(Optional) Azure CV Indexing** – Optionally indexes the video frames using Computer Vision for advanced content-based search.

## **2. Video Agent**

**VideoAgent** is optimized for deep video understanding and question answering using the **Multi-Modal Critical Thinking (MMCT)** framework.

[![](/docs/multimedia/videoAgent.webp)](https://arxiv.org/pdf/2405.18358)

The agent uses the **Multi-Modal Critical Thinking (MMCT)** framework ([arxiv.org/abs/2405.18358](https://arxiv.org/abs/2405.18358)) to generate high-quality answers through structured reasoning. MMCT involves two key components:

   - **Planner**: Drives the reasoning process using a coordinated toolchain, generating an initial response based on video analysis.
   - **Critic (optional)**: Evaluates the planner's output and provides feedback to improve accuracy and decision-making.

> **Note:** The critic agent is enabled by default. You can disable it by setting `use_critic_agent=False` during initialization.
> **Disabling the critic agent skips the critical thinking feedback loop and may reduce the accuracy of the final response.**

---

## **Tool Workflow**

**VideoAgent uses a fixed toolchain** that works collaboratively during the video question answering process. These tools are automatically orchestrated by the planner:

- `GET_VIDEO_SUMMARY` – Retrieves the most relevant video for the query, along with its summary.
- `GET_OBJECT_COLLECTION` – Retrieves the most relevant video for the query, along with its detected objects.
- `GET_CONTEXT` – Extracts relevant transcript and visual summary chunks related to the query.
- `GET_RELEVANT_FRAMES` – Provides semantically similar keyframes related to the query using CLIP embeddings.
- `QUERY_FRAME` – Queries specific video keyframes to extract detailed visual information and provide additional context to the planner.

The tools work together in a coordinated pipeline to ensure comprehensive video understanding that combines both textual and visual information for accurate question answering.

---

## **Usage**

### **Video Ingestion**

```python
from mmct.video_pipeline import IngestionPipeline, Languages
from mmct.config.providers import IngestionProviderConfig
from mmct.providers.azure import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
    AISearchChapterProvider,
    AISearchKeyframesProvider,
    AISearchObjectCollectionProvider,
    AzureStorageProvider,
    WhisperTranscriptionProvider
)
from mmct.providers.local import ClipImageEmbeddingProvider
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential

# Configure credentials (or use api_key directly)
credentials = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

# Initialize the provider
provider = IngestionProviderConfig(
    llm_provider=AzureLLMProvider(
        endpoint="https://<your-openai-endpoint>.openai.azure.com/",
        deployment_name="<your-llm-deployment-name>",
        model_name="<your-llm-model-name>",
        api_version="<your-api-version>",
        credentials=credentials,
    ),
    embedding_provider=AzureEmbeddingProvider(
        endpoint="https://<your-openai-endpoint>.openai.azure.com/",
        deployment_name="<your-embedding-deployment-name>",
        api_version="<your-api-version>",
        credentials=credentials,
    ),
    image_embedding_provider=ClipImageEmbeddingProvider(),
    vectordb_chapter=AISearchChapterProvider(
        endpoint="https://<your-search-service>.search.windows.net",
        index_name="<your-chapter-index-name>",
        credentials=credentials,
    ),
    vectordb_keyframes=AISearchKeyframesProvider(
        endpoint="https://<your-search-service>.search.windows.net",
        index_name="<your-keyframe-index-name>",
        credentials=credentials,
    ),
    vectordb_object_registry=AISearchObjectCollectionProvider(
        endpoint="https://<your-search-service>.search.windows.net",
        index_name="<your-object-registry-index-name>",
        credentials=credentials,
    ),
    storage_provider=AzureStorageProvider(
        storage_account_name="<your-storage-account-name>",
        keyframe_container_name="<your-keyframe-container-name>",
        credentials=credentials,
    ),
    transcription_provider=WhisperTranscriptionProvider(
        endpoint="https://<your-openai-endpoint>.openai.azure.com/",
        api_version="<your-api-version>",
        deployment_name="<your-whisper-deployment-name>",
        credentials=credentials,
    ),
)

ingestion = IngestionPipeline(
    video_path="path/to/your/video.mp4",
    language=Languages.ENGLISH_INDIA,
    provider=provider
)

# Run the ingestion pipeline
await ingestion.run()
```

### **Video Agent**

```python
from mmct.video_pipeline import VideoAgent
from mmct.config.providers import VideoAgentProviderConfig
from mmct.providers.azure import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
    AISearchChapterProvider,
    AISearchKeyframesProvider,
    AISearchObjectCollectionProvider,
    AzureStorageProvider
)
from mmct.providers.local import ClipImageEmbeddingProvider
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential

# Configure credentials (or use api_key directly)
credentials = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

# Initialize the provider
provider = VideoAgentProviderConfig(
    llm_provider=AzureLLMProvider(
        endpoint="https://<your-openai-endpoint>.openai.azure.com/",
        deployment_name="<your-llm-deployment-name>",
        model_name="<your-llm-model-name>",
        api_version="<your-api-version>",
        credentials=credentials,
    ),
    embedding_provider=AzureEmbeddingProvider(
        endpoint="https://<your-openai-endpoint>.openai.azure.com/",
        deployment_name="<your-embedding-deployment>",
        api_version="<your-api-version>",
        credentials=credentials,
    ),
    image_embedding_provider=ClipImageEmbeddingProvider(),
    vectordb_chapter=AISearchChapterProvider(
        endpoint="https://<your-search-service>.search.windows.net",
        index_name="<your-chapter-index-name>",
        credentials=credentials,
    ),
    vectordb_keyframes=AISearchKeyframesProvider(
        endpoint="https://<your-search-service>.search.windows.net",
        index_name="<your-keyframe-index-name>",
        credentials=credentials,
    ),
    vectordb_object_registry=AISearchObjectCollectionProvider(
        endpoint="https://<your-search-service>.search.windows.net",
        index_name="<your-object-registry-index-name>",
        credentials=credentials,
    ),
    storage_provider=AzureStorageProvider(
        storage_account_name="<your-storage-account-name>",
        keyframe_container_name="<your-keyframe-container-name>",
        credentials=credentials,
    )
)

# Configure the Video Agent
video_agent = VideoAgent(
    query="Your question about the video",
    video_id=None,  # Optional: specify video ID
    url=None,  # Optional: URL for filtering
    use_critic_agent=True,  # Enable critical thinking framework
    stream=False,  # Stream logs of the agentic flow
    cache=False,  # Optional: enable caching
    provider=provider
)

# Execute video analysis
response = await video_agent()
print(response.response)
```

### **Using Custom LLM Providers**

You can implement custom LLM providers for any vendor (Anthropic, Cohere, etc.) by inheriting from `BaseLLMProvider`:

```python
from mmct.providers.base import BaseLLMProvider
from mmct.config.providers import VideoAgentProviderConfig, IngestionProviderConfig
from mmct.video_pipeline import VideoAgent, IngestionPipeline

# Your custom LLM provider implementation
class CustomLLMProvider(BaseLLMProvider):
    # Implement required abstract methods: chat_completion() and get_autogen_client()
    pass

# Use your custom provider with VideoAgent
custom_llm = CustomLLMProvider(api_key="your-api-key", model_name="your-model")
provider = VideoAgentProviderConfig(
    llm_provider=custom_llm,
    # ... other required providers
)

video_agent = VideoAgent(query="Your query", provider=provider)
response = await video_agent()
```

For a complete working example of a custom Anthropic provider, see [`examples/image_agent.ipynb`](../../examples/image_agent.ipynb).

For detailed implementation instructions, refer to the [Providers Guide](../providers/README.md).

---
