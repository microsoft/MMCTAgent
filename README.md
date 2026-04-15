<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.18358-b31b1b.svg)](https://arxiv.org/abs/2405.18358)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

# [**MMCTAgent**](https://arxiv.org/abs/2405.18358)

<div align="center">
  <strong>Multi-Modal Critical Thinking Agent Framework for Complex Visual Reasoning</strong>
  <br><br>
  <a href="https://youtu.be/Lxt1b_U-a68">🎥 Demo Video</a> • 
  <a href="https://arxiv.org/abs/2405.18358">📄 Research Paper</a> • 
  <a href="#getting-started">🚀 Quick Start</a>
</div>

<br>

<div align="center">

![Demo GIF](docs/multimedia/gif/Demo_MMCT.gif)

**▶️ [Watch Demo Video](https://youtu.be/Lxt1b_U-a68)**

</div>

## Overview

MMCTAgent is a state-of-the-art multi-modal AI framework that brings human-like critical thinking to visual reasoning tasks. it combines advanced planning, self-critique, and tool-based reasoning to deliver superior performance in complex image and video understanding applications.

### Why MMCTAgent?

- **🧠 Self Reflection Framework**: MMCTAgent emulates iteratively analyzing multi-modal information, decomposing complex queries, planning strategies, and dynamically evolving its reasoning. Designed as a research framework, MMCTAgent integrates critical thinking elements such as verification of final answers and self-reflection through a novel approach that defines a vision-based critic and identifies task-specific evaluation criteria, thereby enhancing its decision-making abilities.
- **🔬 Enables Querying over Multimodal Collections**: It enables modular design to plug-in right audio, visual extraction and processing tools, combined with Multimodal LLMs to ingest and query over large number of videos and image data.
- **🚀 Easy Integration**: Its modular design allows for easy integration into existing workflows and adding domain-specific tools, facilitating adoption across various domains requiring advanced visual reasoning capabilities.


### 🤖 Graph Agent (Swarm)

<p align="center">
  <img src="docs/multimedia/media_src/Graph%20Agent.jpg" alt="Graph Agent architecture" width="95%" />
</p>

### 🔁 Graph State (Deterministic Pipeline)

<p align="center">
  <img src="docs/multimedia/media_src/Graph%20State.jpg" alt="Graph State pipeline" width="95%" />
</p>

<details>
<summary><strong>📥 Ingestion Pipeline</strong></summary>
<br>
<p align="center">
  <img src="docs/multimedia/media_src/Ingestion%20Pipeline.jpg" alt="Ingestion Pipeline" width="95%" />
</p>
</details>

## **Key Features**

- **📊 Dual Orchestration for Video**:
  - **Graph Agent (Swarm)**: An autonomous multi-agent team (Planner, Video, Image) that agentically traverses knowledge graphs for broad discovery and reasoning.
  - **Graph State (Pipeline)**: A deterministic state-machine workflow designed for high-precision, reproducible, and efficient retrieval.
- **🖼️ Agentic Image Reasoning**: A dedicated Multi-Agent Image Pipeline featuring specialized tools for OCR, Object Detection, Scene Recognition, and ViT-based reasoning.
- **📂 Structured Video Ingestion**: End-to-end pipeline to transform raw video into a queryable temporal knowledge graph (Neo4j) with automated scene segmentation and metadata extraction.
- **🧠 Provider Agnostic**: Native support for Azure OpenAI, Neo4j, and Azure Blob Storage, with a modular interface for custom LLM and VectorDB backends.
- **🔌 MCP Native**: Built-in Model Context Protocol (MCP) server support for seamless integration with external AI clients.

---

## **Getting Started**

### **Installation**

1. To get started with MMCTAgent, clone the repository and install the dependencies:

   ```bash
   git clone https://github.com/microsoft/MMCTAgent.git
   cd MMCTAgent
   ```

2. **System Dependencies**
    
   Install FFmpeg

   **Linux/Ubuntu:**
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg libsm6 libxext6 -y
   ```
   
   **Windows:**
   - Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add the `bin` folder to your system PATH

3. **Python Environment Setup**

   **Option A: Using Conda (Recommended)**
   ```bash
   conda create -n mmct-agent python=3.11
   conda activate mmct-agent
   ```

   **Option B: Using venv**
   ```bash
   python -m venv mmct-agent
   # Linux/Mac
   source mmct-agent/bin/activate
   # Windows
   mmct-agent\Scripts\activate.bat
   ```

4. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install .
   ```

5. **Quick Start Examples**

#### **1. Video Ingestion**

Transform a video file into a structured knowledge graph (Neo4j).

```python
import asyncio
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import IngestionPipeline
from mmct.video_pipeline.core.ingestion.languages import Languages
# Import your provider configuration or use defaults
from config.provider_config import get_ingestion_providers

async def run_ingestion():
    providers = get_ingestion_providers()
    
    ingestion = IngestionPipeline(
        video_path="path/to/video.mp4",
        video_id="my-unique-video-id",
        language=Languages.ENGLISH_INDIA,
        provider=providers
    )

    report = await ingestion.run()
    print(f"Ingestion {report.status}: {report.summary()}")

asyncio.run(run_ingestion())
```

### **2. Video Q&A (Unified Pipeline)**

Query your video knowledge graph using either agentic swarm or deterministic state modes.

```python
from mmct.video_pipeline.query_pipeline import VideoQueryPipeline, QueryPipelineMode
import asyncio

async def query_video():
    # Initialize the unified pipeline
    # use_provider_defaults=True automatically hydrates providers from config
    pipeline = VideoQueryPipeline(
        mode=QueryPipelineMode.GRAPH_STATE,  # or GRAPH_AGENT for swarm
        use_provider_defaults=True,
        use_critic=True
    )

    # Perform a natural language query
    result = await pipeline.query(
        user_query="What happens after the car enters the frame?",
        video_id="my-unique-video-id"
    )

    print(f"Answer: {result['answer']}")
    await pipeline.close()

asyncio.run(query_video())
```

### **3. Image Q&A**

Analyze individual images using specialized vision tools.

```python
from mmct.image_pipeline.agents.image_agent import ImageAgent, ImageQnaTools
import asyncio

async def query_image():
    agent = ImageAgent(
        image_path="path/to/image.jpg",
        query="What labels are on the medicine bottles?",
        use_critic_agent=True,
        tools=[ImageQnaTools.ocr, ImageQnaTools.vit]
    )

    response = await agent()
    print(f"Analysis: {response}")

asyncio.run(query_image())
```

### **4. MCP Server**

Expose MMCT pipelines as tools for model-based agents.

```bash
# Start the server
PYTHONPATH=. python3 mcp_server/main.py
```
*Access the server at `http://0.0.0.0:8000/mcp`. See the [`mcp_server/`](mcp_server/) directory for more details.*

---

## **Provider System**

### **Multi-Cloud & Vendor-Agnostic Architecture**

MMCTAgent features a modular provider system that allows you to switch between different cloud providers and AI services seamlessly.

| Service Type | Supported Providers | Use Cases |
|--------------|--------------------|-----------|
| **LLM** | Azure OpenAI **+ Custom** | Reasoning, chat completion, planning |
| **Search/DB** | Neo4j | Graph traversal, vector search |
| **Transcription** | Azure Speech, OpenAI Whisper | Audio-to-text conversion |
| **Storage** | Azure Blob Storage, Local | Media and artifact management |

> **Note**: For detailed implementation, see the [Providers Guide](mmct/providers/README.md).

---

## **Project Structure**

```sh
MMCTAgent
├── examples/               # 💡 Comprehensive Jupyter notebook examples
├── mcp_server/             # 🔌 Model Context Protocol server entry points
├── mmct/                   # 🧠 Core Framework
│   ├── image_pipeline/     # Multi-Agent image reasoning and vision tools
│   ├── video_pipeline/     # End-to-end video intelligence
│   │   ├── core/           # Ingestion, Graph structure, and Registry
│   │   ├── graph_agent/    # Swarm-based (Agentic) orchestration
│   │   ├── graph_state/    # State-Machine (Deterministic) orchestration
│   │   └── query_pipeline.py # Unified video query entry point
│   ├── providers/          # Modular backend service interfaces
│   └── utils/              # Error handling and shared utilities
├── api/                    # 🌐 FastAPI web application
├── config/                 # ⚙️ Centralized configuration and hydration
├── infra/                  # 🏗️ Azure Infrastructure deployment guides
└── README.md
```

---

## **Contributing**

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact opencode@microsoft.com with any additional questions or comments.


> *Note:* This project is currently under active research and continuous development. While contributions are encouraged, please note that the codebase may evolve as the project matures.

## **Citation**

If you find MMCTAgent useful in your research, please cite:

```bibtex
@article{kumar2024mmctagent,
  title={MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning},
  author={Kumar, Somnath and Gadhia, Yash and Ganu, Tanuja and Nambi, Akshay},
  conference={NeurIPS OWA-2024},
  year={2024},
  url={https://www.microsoft.com/en-us/research/publication/mmctagent-multi-modal-critical-thinking-agent-framework-for-complex-visual-reasoning}
}
```

## **License**

Licensed under the [MIT License](LICENSE).

---

## **Support**

- [Documentation](docs/)
- [Report Issues](https://github.com/microsoft/MMCTAgent/issues)
- [Discussions](https://github.com/microsoft/MMCTAgent/discussions)
