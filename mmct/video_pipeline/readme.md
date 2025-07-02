# **MMCT - Video Pipeline**

[![](/docs/multimedia/videoPipeline.webp)](https://arxiv.org/pdf/2405.18358)

## **Overview**

Video Pipeline consists of two components:

1. Video Ingestion
2. Video Agent.

You can first ingest your specific video using the Video Ingestion pipeline, which processes and prepares the video content for downstream tasks. Once ingested, the **Video Agent** can be used to perform Question Answering over the video. It leverages multiple tools to accurately extract relevant information and optionally uses a **Critic Agent** (if enabled) to enhance the quality and precision of the response.

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
7. **(Optional) Azure CV Indexing** – Optionally indexes the video frames using Azure Computer Vision for advanced content-based search.

## **2. Video Agent**

**VideoAgent** operates in two key stages:

1. **Video Retrieval** – Given a user query, the agent retrieves relevant videos from a pre-ingested **Azure AI Search index**. This search ensures that only contextually relevant videos are passed on for deep analysis.

2. **Video Question Answering (QA)** 

[![](/docs/multimedia/videoAgent.webp)](https://arxiv.org/pdf/2405.18358)

After retrieval, the agent uses the **Multi-Modal Critical Thinking (MMCT)** framework ([arxiv.org/abs/2405.18358](https://arxiv.org/abs/2405.18358)) to generate a high-quality answer. MMCT involves two agents:

   - **Planner**: Drives the reasoning process using a structured toolchain, generating an initial response.
   - **Critic (optional)**: Analyzes the planner’s output and, if needed, provides feedback that prompts an improved final answer.

> **Note:** The critic agent is enabled by default. You can disable it by setting `use_critic_agent=False` during initialization.  
> **Disabling the critic agent skips the feedback loop and may reduce the accuracy of the final response.**

---

## **Tool Workflow**

Unlike independent tool selection, **VideoAgent uses a fixed pipeline** of tools that work collaboratively during the QA stage. These tools are automatically orchestrated by the planner:

- `GET_SUMMARY_TRANSCRIPT` – Extracts the full transcript and a high-level visual summary of the video.
- `QUERY_SUMMARY_TRANSCRIPT` – Given a query, this tool identifies **three timestamps** in the transcript that are most relevant.
- `QUERY_AZURE_COMPUTER_VISION` _(optional)_ – Uses **Azure Computer Vision** to return **three additional timestamps** related to the visual content of the query.
- `QUERY_GPT4V` – Uses **OpenAI's GPT-4V** to inspect video frames around the identified timestamps and generate a detailed response grounded in both visual and textual understanding.

By default, all tools are used in a coordinated pipeline. You can disable **only** the Azure Computer Vision tool by setting `use_azure_cv_tool=False` during agent initialization.

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
import ast
from mmct.video_pipeline import VideoAgent

query = ""
index_name = ""  # Azure AI Search index name
top_n = 3  # Number of top results (video ids for MMCT VQnA) to return from the index
use_azure_cv_tool = False   # flag for selection of Azure Computer Vision Tool
use_critic_agent = True     # flag to utilize Critic Agent.
stream = True               # flag to stream the logs of the Agentic Flow.

video_agent = VideoAgent(
    query=query,
    index_name=index_name,
    top_n=top_n,
    use_azure_cv_tool=use_azure_cv_tool,
    use_critic_agent=use_critic_agent,
    stream=stream,
)

response = asyncio.run(video_agent())
print(response.response)
```
