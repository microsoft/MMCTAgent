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
  
[![Demo Video](https://img.youtube.com/vi/Lxt1b_U-a68/maxresdefault.jpg)](https://youtu.be/Lxt1b_U-a68)

**▶️ [Watch Demo Video](https://youtu.be/Lxt1b_U-a68)**

</div>


## Overview

MMCTAgent is a state-of-the-art multi-modal AI framework that brings human-like critical thinking to visual reasoning tasks. it combines advanced planning, self-critique, and tool-based reasoning to deliver superior performance in complex image and video understanding applications.

### Why MMCTAgent?

- **🧠 Human-like Reasoning**: Implements structured critical thinking with planning and critique phases
- **🎯 Superior Performance**: Outperforms traditional approaches on complex visual reasoning benchmarks  
- **🔧 Production Ready**: Enterprise-grade architecture with multi-cloud support and security features
- **🚀 Easy Integration**: Modular design allows seamless integration into existing workflows

<p align="center">
  <a href="https://arxiv.org/abs/2405.18358">
    <img src="docs/multimedia/VideoPipeline.webp" alt="Video Pipeline - Main Architecture" width="80%" />
  </a>
</p>

## **Key Features**

### Critical Thinking Architecture

MMCTAgent is inspired by human cognitive processes and integrates a structured reasoning loop:

- **Planner**:  
  Generates an initial response using relevant tools for visual or multi-modal inputs.

- **Critic** *(optional)*:  
  Evaluates the Planner’s response and provides feedback to improve accuracy and decision-making.  
  > The Critic is enabled by default. To disable, set `use_critic_agent=False`.

---

### Modular Agents

MMCTAgent includes two specialized agents:

#### ImageAgent

[![](docs/multimedia/imageAgent.webp)](https://arxiv.org/abs/2405.18358)

A reasoning engine tailored for static image understanding.  
It supports a configurable set of tools via the `ImageQnaTools` enum:

- `OBJECT_DETECTION` – Detects objects in the image.
- `OCR` – Extracts embedded text content.
- `RECOG` – Recognizes scenes, faces, or objects.
- `VIT` – Applies GPT-4V for high-level visual reasoning.

> The Critic can be toggled via `use_critic_agent` flag.

---

#### VideoAgent
[![](docs/multimedia/videoPipeline.webp)](https://arxiv.org/abs/2405.18358)

Optimized for deep video understanding through a structured two-stage pipeline:

1. **Video Retrieval**  
   Uses an Azure AI Search index to fetch videos relevant to a user query.

2. **Video Question Answering**  

[![](docs/multimedia/videoAgent.webp)](https://arxiv.org/abs/2405.18358)

   Applies a fixed toolchain orchestrated by the Planner:

   - `GET_VIDEO_DESCRIPTION` – Extracts transcript and visual summary.
   - `QUERY_VIDEO_DESCRIPTION` – Finds top-3 relevant timestamps.
   - `QUERY_FRAMES_COMPUTER_VISION` *(optional)* – Identifies visual cues.
   - `QUERY_VISION_LLM` – Combines frame-level visual and textual analysis.

> The Critic agent helps validate and refine answers, improving reasoning depth.

For more details, refer to the full research article:

**[MMCTAgent: Multi-modal Critical Thinking Agent
 Framework for Complex Visual Reasoning](https://arxiv.org/abs/2405.18358)**  
Published on **arXiv** – [arxiv.org/abs/2405.18358](https://arxiv.org/abs/2405.18358)

## Citation

If you find MMCTAgent useful in your research, please cite our paper:

```bibtex
@article{kumar2024mmctagent,
  title={MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning},
  author={Kumar, Somnath and Gadhia, Yash and Ganu, Tanuja and Nambi, Akshay},
  journal={arXiv preprint arXiv:2405.18358},
  year={2024},
  url={https://arxiv.org/abs/2405.18358}
}
```

---

## **Table of Contents**

- [Provider System](#provider-system)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## **Provider System**

### Multi-Cloud & Vendor-Agnostic Architecture

MMCTAgent now features a **modular provider system** that allows you to seamlessly switch between different cloud providers and AI services without changing your application code. This makes the framework truly **vendor-agnostic** and suitable for various deployment scenarios.

#### Supported Providers

| Service Type | Supported Providers | Use Cases |
|--------------|--------------------|-----------|
| **LLM** | Azure OpenAI, OpenAI | Text generation, chat completion |
| **Search** | Azure AI Search | Document search and retrieval |
| **Vision** | Azure Computer Vision, OpenAI Vision | Image analysis, object detection |
| **Transcription** | Azure Speech Services, OpenAI Whisper | Audio-to-text conversion |
| **Storage** | Azure Blob Storage, Local Storage | File storage and management |

#### Key Benefits

- **🔄 Vendor Independence**: Switch between Azure, OpenAI, and other providers
- **🛡️ Enhanced Security**: Built-in support for Managed Identity and Key Vault
- **⚙️ Flexible Configuration**: Environment-based or programmatic configuration
- **🔧 Easy Migration**: Backward compatibility with existing configurations
- **📊 Centralized Management**: Single configuration point for all services

#### Quick Provider Configuration

```bash
# Azure-first setup
LLM_PROVIDER=azure
SEARCH_PROVIDER=azure_ai_search
VISION_PROVIDER=azure

# OpenAI-first setup  
LLM_PROVIDER=openai
VISION_PROVIDER=openai
TRANSCRIPTION_PROVIDER=openai

# Hybrid setup
LLM_PROVIDER=azure
SEARCH_PROVIDER=elasticsearch
VISION_PROVIDER=openai
```

For detailed configuration instructions, see our [Provider Configuration Guide](docs/PROVIDERS.md).

---

## Getting Started

### Quick Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/microsoft/MMCTAgent.git
   cd MMCTAgent
   ```

2. **System Dependencies**
   
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
   pip install -r requirements.txt
   ```

### Verify Installation
```python
from mmct.image_pipeline import ImageAgent
print("✅ MMCTAgent installed successfully!")
```

## **Prerequisites**

Below are the Azure Resources that are required to execute this repository. You can checkout the `infra` folder and utilize the `INFRA_DEPLOYMENT_GUIDE` to not only deploy the resources through ARM Templates but also build the containers and directly deploy the script to Azure App Services and Azure Container Apps.

| Resource Name                 | Documentation Article | Microsoft Intra-Identity Role |
|--------------------------------|----------------------|------------------------------------|
| Storage Account                | [Document](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-overview)        | *Storage Blob Data Reader/Contributor* |
| Azure Computer Vision  [Optional]        | [Document](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/)        | *Cognitive Services User*           |
| Azure OpenAI (4o, 4o-mini, text-embedding-ada-002, Whisper) | [Document](https://learn.microsoft.com/en-us/azure/ai-services/openai/) | *Cognitive Services OpenAI User* |
| Azure AI Search                | [Document](https://learn.microsoft.com/en-us/azure/search/)        | *Search Index Data Contributor*       |
| Azure AI Search                | [Document](https://learn.microsoft.com/en-us/azure/search/)        | *Search Service Contributor*       |
| Azure Speech Service           | [Document](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)        | *Cognitive Services Speech Contributor* or *Cognitive Services Speech User* role.          |
| Azure App Service [Optional] | [Document](https://learn.microsoft.com/en-us/azure/app-service/)        | *NA*             |
| Azure Event Hub [Optional] | [Document](https://learn.microsoft.com/en-us/azure/app-service/)        | *Azure Event Hubs Data Owner* |
| Azure Container Registry [Optional] | [Document](https://learn.microsoft.com/en-us/azure/container-registry/) | *Reader or Contributor* |
| Application Insights [Optional]          | [Document](https://learn.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview)        | *NA*                                |

> Note: If you want to utilize the Microsoft Azure Intra Id Access then you can assign the above corresponding roles for the each resource. Otherwise you can use the API Key or Connection String approach to utilize the resources.

## **Configuration**

### Environment Setup

MMCTAgent uses a flexible configuration system that supports multiple cloud providers. Choose your configuration method:

#### Quick Start - Copy Environment Template

```bash
# For development
cp config/environments/development.env .env

# For production
cp config/environments/production.env .env
```

Then edit `.env` with your specific values.

#### Provider Configuration Examples

**Azure-First Setup:**
```bash
# LLM Configuration
LLM_PROVIDER=azure
LLM_ENDPOINT=https://your-resource.openai.azure.com/
LLM_DEPLOYMENT_NAME=gpt-4o
LLM_MODEL_NAME=gpt-4o
LLM_USE_MANAGED_IDENTITY=true

# Search Configuration
SEARCH_PROVIDER=azure_ai_search
SEARCH_ENDPOINT=https://your-search.search.windows.net
SEARCH_USE_MANAGED_IDENTITY=true
SEARCH_INDEX_NAME=your-index-name

# Storage Configuration
STORAGE_PROVIDER=azure_blob
STORAGE_ACCOUNT_NAME=your-storage-account
STORAGE_USE_MANAGED_IDENTITY=true
```

**OpenAI Setup:**
```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_ENDPOINT=https://api.openai.com
LLM_MODEL_NAME=gpt-4o
OPENAI_API_KEY=your-openai-api-key

# Vision Configuration
VISION_PROVIDER=openai
OPENAI_VISION_MODEL=gpt-4o

# Transcription Configuration
TRANSCRIPTION_PROVIDER=openai
OPENAI_WHISPER_MODEL=whisper-1
```

**Hybrid Setup:**
```bash
# Use Azure for LLM
LLM_PROVIDER=azure
LLM_ENDPOINT=https://your-resource.openai.azure.com/

# Use OpenAI for vision
VISION_PROVIDER=openai
OPENAI_API_KEY=your-openai-key

# Use Elasticsearch for search
SEARCH_PROVIDER=elasticsearch
ELASTICSEARCH_ENDPOINT=https://your-elasticsearch.com
```

### Security Configuration

#### Managed Identity (Recommended for Azure)
```bash
LLM_USE_MANAGED_IDENTITY=true
SEARCH_USE_MANAGED_IDENTITY=true
STORAGE_USE_MANAGED_IDENTITY=true
```

#### Azure Key Vault (Production)
```bash
ENABLE_SECRETS_MANAGER=true
KEYVAULT_URL=https://your-keyvault.vault.azure.net/
```

### Logging Configuration
```bash
LOG_LEVEL=INFO
LOG_ENABLE_FILE=true
LOG_ENABLE_JSON=false
LOG_MAX_FILE_SIZE=10 MB
```

📖 **For comprehensive configuration options, see our [Provider Configuration Guide](docs/PROVIDERS.md)**

## Usage

### Quick Start Examples

#### Image Analysis with MMCTAgent

```python
from mmct.image_pipeline import ImageAgent, ImageQnaTools
import asyncio

# Initialize the Image Agent with desired tools
image_agent = ImageAgent(
    query="What objects are visible in this image and what text can you read?",
    image_path="path/to/your/image.jpg",
    tools=[ImageQnaTools.OBJECT_DETECTION, ImageQnaTools.OCR, ImageQnaTools.VIT],
    use_critic_agent=True,  # Enable critical thinking
    stream=False
)

# Run the analysis
response = asyncio.run(image_agent())
print(f"Analysis Result: {response.response}")
```

#### Video Understanding with MMCTAgent

```python
from mmct.video_pipeline import VideoAgent
import asyncio

# Configure the Video Agent
video_agent = VideoAgent(
    query="Explain the main events in this video",
    index_name="your-search-index",  # Azure AI Search index
    top_n=3,  # Number of relevant videos to analyze
    use_computer_vision_tool=True,   # Enable visual analysis
    use_critic_agent=True,           # Enable critical review
    stream=True                      # Stream progress logs
)

# Execute video analysis
response = asyncio.run(video_agent())
print(f"Video Analysis: {response}")
```

For more comprehensive examples, see the [`examples/`](examples/) directory.

## **Project Structure**

Below is the project structure highlighting the key entry-point scripts for running the three main pipelines— `Image QNA`, `Video Ingestion` and `Video Agent`.

```sh
MMCTAgent
| 
├── infra
|   └── INFRA_DEPLOYMENT_GUIDE.md    # Guide for deployment of Azure Infrastructure 
├── app                              # contains the FASTAPI application over the mmct pipelines.
├── mmct
│   ├── .
│   ├── image_pipeline
│   │   ├── agents
│   │   │    └── image_agent.py      #  Entry point for the MMCT Image Agentic Workflow
│   │   └── README.md                #  Guide for Image Pipeline
│   └── video_pipeline
│       ├── agents
│       │   └── video_agent.py      # Entry point for the MMCT Video Agentic Workflow
│       ├── core
│       │     └── ingestion
│       │           └── ingestion_pipeline.py   # Entry point for the Video Ingestion Workflow
│       └── README.md                # Guide for Video Pipeline  
├── requirements.txt
└── README.md  
```

## Contributing

We welcome contributions from the community! MMCTAgent is an open-source project and we encourage you to help make it better.

### How to Contribute

1. **Fork the Repository**: Click the "Fork" button on GitHub
2. **Create a Feature Branch**: `git checkout -b feature/your-feature-name`
3. **Make Changes**: Implement your improvements
4. **Add Tests**: Ensure your changes are well-tested
5. **Submit a Pull Request**: Describe your changes and submit for review

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/MMCTAgent.git
cd MMCTAgent

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Provide clear, detailed descriptions with reproducible examples
- Check existing issues to avoid duplicates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with support from Microsoft Research
- Thanks to all contributors and the open-source community
- Special recognition to the authors of the foundational research

## Support

- 📖 [Documentation](docs/)
- 🐛 [Report Issues](https://github.com/microsoft/MMCTAgent/issues)
- 💬 [Discussions](https://github.com/microsoft/MMCTAgent/discussions)
- 📧 Contact: [Research Team](mailto:mmctagent@microsoft.com)

---

<div align="center">
  <strong>Made with ❤️ by the MMCTAgent Team</strong>
  <br>
  <a href="https://github.com/microsoft/MMCTAgent">⭐ Star us on GitHub</a>
</div>
