# MMCT - Image Pipeline

This pipeline enables Question Answering (Q/A) on images.
![plot](images/MMCT_autogen.jpg)

## Table of Contents

- [Installation](#installation)
  - [Create Conda Environment](#create-conda-environment)
  - [Install Requirements](#install-requirements)
- [Setting up .env](#setting-up-.env)
- [Running the Pipeline](#running-the-pipeline)
- [Conversation History](#conversation-history)

## Installation

### Create Conda Environment

First, create a new Conda environment with Python 3.11:

```bash
conda create --name <env_name> python==3.11
```

Replace <env_name> with your desired environment name.

Activate the environment:
```bash
conda activate <env_name>
```

### Install Requirements
Navigate to the image_pipeline directory and install the required packages:
```bash
cd image_pipeline
pip install -r requirements.txt
pip install -e .
```

### Setting up .env

.env file is located in `image_pipeline/agents/` directory.

If you are using TRAPI resources, set `TRAPI_RESOURCE_ENABLE` variable to `True`.
```sh
TRAPI_RESOURCE_ENABLE='True'
```
Else add your Azure Resources data to the env variables.
```sh
AZURE_OPENAI_ENDPOINT = ''
AZURE_OPENAI_API_VERSION = ''
AZURE_OPENAI_MODEL = ''
AZURE_OPENAI_MODEL_VERSION = ''
AZURE_OPENAI_VISION_MODEL = ''
AZURE_OPENAI_VISION_MODEL_VERSION = ''
```

### Running the Pipeline
To run the pipeline, use the main.py file located in the agents directory. Ensure you are in the same environment.

Navigate to the agents directory:
```bash
cd ../agents
```

Define your queries in main.py in the queries list within main.py.

Run the script:
```bash
python main.py
```

### Conversation History
After running the pipeline, the conversation history will be saved in output.txt.
