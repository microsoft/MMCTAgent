## Image Pipeline

To run the image pipeline one must first input all credentials into a `.env` folder.
`.env.template` file can be taken as reference to add the credentials.

### Setup

To setup we provide conda environment.yml and requirements.txt, any one of the following can be utilized
```bash
$ conda env create --name mmct -f environment.yml
```
or
```bash
$ pip install -r requirements.txt
```
and
Python version must be 3.8.19

### Scripts

To Evaluate our pipeline one can change the data path and run 

```bash
$ python main_script.py
```
To use Critic based MMCT.

### Overview

While the pipeline is modular with easy control over individal react steps and functionality. The pipeline also allows easy addition of different tools and dataset

A overview of the filestructre will be helpful to modify over the codebase.

```bash
.
├── data       ## Dataset
│   ├── __init__.py
│   ├── mm_vet
│   │   ├── bard_set.json
│   │   ├── dataloader.py
│   │   ├── __init__.py
│   │   └── mm-vet.json
│   └── nlvr
│       ├── __init__.py
│       ├── nlvr
│       │   └── ...
│       ├── nlvr2
│       │   └── ...
│       └── README.md
├── env        ## Env is a gym env to have finegrain control over the iteration
│   ├── blip_react_env.py
│   ├── __init__.py
│   ├── react_env
│   │   └── react_env.py
│   └── Tool_env.py
├── __init__.py
├── main_cot_only_tools.py                  ## Scripts
├── main_script.py                          ## Main Script
├── main_old.py                             ## Scripts
├── models                  ## Model definition under each category
│   ├── __init__.py
│   ├── LLMs
│   │   ├── base_llm.py
│   │   ├── llama.py
│   │   ├── lora.py
│   │   └── utils.py
│   └── VIT
│       ├── beit3.py
│       ├── beit3.spm
│       ├── gpt4v.py
│       ├── instructblip.py
│       ├── object_detect
│       │   ├── deta_res.py
│       │   ├── deta_swinL.py
│       │   ├── __init__.py
│       │   ├── yolov8s.pt
│       │   └── yolov8s.py
│       ├── ocr
│       │   ├── __init__.py
│       │   ├── trocr_base.py
│       │   ├── trocr_large.py
│       │   └── trocr_small.py
│       └── recog
│           ├── __init__.py
│           ├── instructBlipCap.py
│           ├── mplug_base.py
│           └── mplug_large.py
├── pipeline                           ## Different Pipeline for controlling Flow
│   ├── basic_cot.py
│   ├── __init__.py
│   └── react.py
├── README.md
├── setup.py                           ## For Installation
├── tool           ## Tools are either API or model calls adapted for the pipeline
│   ├── azure
│   │   ├── imun.py
│   │   ├── toolkit.py
│   │   ├── tool.py
│   │   └── utils.py
│   ├── base.py
│   ├── blip.py
│   ├── critic.py
│   ├── __init__.py
│   ├── object_detect.py
│   ├── ocr.py
│   ├── recog.py
│   └── vit.py
└── utils                               ## Auxilary
    ├── ds_utils.py
    └── logger.py
```

