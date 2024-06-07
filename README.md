<div align="center">

![](https://img.shields.io/badge/Task-Code_Related-blue)
![](https://img.shields.io/badge/Code_License-MIT-green)

</div>

Here's our paper on arxiv: [link to paper](https://arxiv.org/abs/2405.18358) 

## MMCTAgent

Code along the submission of "MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning".

The repository majorly contains two pipelines 
```bash
MMCTAgent
├── video_pipeline
│   ├─  ...
│   └── README.md
│
└── image_pipeline
    ├─  ...
    └── README.md
```

Description about each pipeline and instruction can be found in their respecitive subfolders
Readme:
 - #### [Video Pipeline](video_pipeline/README.md)
 - #### [Image Pipeline](image_pipeline/README.md)

## System Setting

While most experiments are based on OpenAI Key, For few inferences and experiments we required GPU. System configuration for the 
Virtual Machine Utilized

| GPU | CPU | RAM |
| -- | -- | -- |
| 1 x A100 80 GB| 64 cpu cores at 3.2GHz| 512 GB  RAM.|


## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@article{MMCT Agent,
  title={MMCTAgent: MMCTAgent is a novel multi-modal critical thinking agent framework designed to address the inherent limitations of current Multimodal LLMs in complex visual reasoning tasks. MMCTAgent iteratively analyzes multi-modal information, decomposes queries, plans strategies, and dynamically evolves its reasoning. Additionally, MMCTAgent incorporates critical thinking elements such as verification of final answers and self-reflection through a novel approach that defines a vision-based critic and identifies task-specific evaluation criteria, thereby enhancing its decision-making abilities.},
  author={Somnath Kumar, Yash Gadhia, Tanuja Ganu, Akshay Nambi},
  journal={arXiv preprint arXiv:2405.18358v1},
  year={2024}
}
```

## 🍀 Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

Resources:

- [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
- [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
- Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns