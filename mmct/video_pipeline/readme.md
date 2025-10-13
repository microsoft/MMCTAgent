# **MMCT - Video Pipeline**

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.18358-b31b1b.svg)](https://arxiv.org/abs/2405.18358)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2405.18358">
    <img src="/docs/multimedia/VideoPipeline.webp" alt="Video Pipeline - Main Architecture" width="70%" />
  </a>
</p>

## **Overview**

MMCTAgent's Video Pipeline is a state-of-the-art multi-modal AI framework that brings human-like critical thinking to video understanding tasks. It consists of two main components:

1. **Video Ingestion** - Comprehensive video processing and preparation
2. **Video Agent** - Multi-modal critical thinking for video question answering

The pipeline processes videos through ingestion, then applies the **Multi-Modal Critical Thinking (MMCT)** framework for sophisticated video understanding and question answering. The **Video Agent** leverages structured reasoning with planner and critic components to deliver superior performance in complex video analysis tasks.

## **1. Video Ingestion**
[![](/docs/multimedia/ingestionpipeline.png)](https://arxiv.org/pdf/2405.18358)
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

- `GET_CONTEXT` – Extracts relevant transcript and visual summary chunks related to the query.
- `GET_RELEVANT_FRAMES` – Provides semantically similar keyframes related to the query using CLIP embeddings.
- `QUERY_FRAME` – Queries specific video keyframes to extract detailed visual information and provide additional context to the planner.

The tools work together in a coordinated pipeline to ensure comprehensive video understanding that combines both textual and visual information for accurate question answering.

---

## **Usage**

> MMCT Video Ingestion

```python
import asyncio
from mmct.video_pipeline import IngestionPipeline, Languages, TranscriptionServices
video_path = ""   # provide the video path
index = ""        # provide the AI Search Index Name
source_language = Languages.ENGLISH_INDIA   # select the valid language
ingestion = IngestionPipeline(
    video_path=video_path,
    index_name=index,
    transcription_service=TranscriptionServices.AZURE_STT, # select the transcription option
    language=source_language,
)

asyncio.run(ingestion())
```

> MMCT Video Agent

```python
import asyncio
from mmct.video_pipeline import VideoAgent

query = "Your question about the video"
index_name = "your-azure-search-index"  # Azure AI Search index name
video_id = None  # Optional: specify specific video ID
url = None  # Optional: URL for analysis
use_critic_agent = True  # Enable critical thinking framework
stream = False  # Flag to stream the logs of the Agentic Flow
use_graph_rag = False  # Optional: use graph RAG
cache = False  # Optional: enable caching

video_agent = VideoAgent(
    query=query,
    index_name=index_name,
    video_id=video_id,
    url=url,
    use_critic_agent=use_critic_agent,
    stream=stream,
    use_graph_rag=use_graph_rag,
    cache=cache
)

response = asyncio.run(video_agent())
print(response.response)
```
