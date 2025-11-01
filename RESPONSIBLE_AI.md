# MMCT Agent: Responsible AI FAQ

## **What is MMCT Agent?**

MMCTAgent is a multi-modal critical thinking agent framework designed to address the limitations of current Multimodal LLMs in visual reasoning tasks. MMCTAgent iteratively analyzes multi-modal information, decomposes queries, plans strategies, and dynamically evolves its reasoning. Additionally, MMCTAgent incorporates critical thinking elements such as verification of final answers and self-reflection through a novel approach that defines a vision-based critic and identifies task-specific evaluation criteria, thereby enhancing its decision-making abilities.

**Input and Output:**
- **Input:** Multimedia objects like video/image/transcript for audio associated with video and a question that has to be answered referring to the provided multimedia object
- **Output:** The answer to the asked question

## **What can MMCT Agent do?**

- MMCT Agent can provide answers to questions whose answer can be found in video/image/audio-transcript
- Its more accurate than directly asking a multimodal-LLM or Azure-Vision service

## **What is/are MMCT Agent's intended use(s)?**

Q&A over multimodal resources like video/image/audio-transcript

## **How was MMCT Agent evaluated? What metrics are used to measure performance?**

- Evaluations was conducted across image and video understanding benchmarks in a zero-shot setting
- Evaluation metric for all datasets is the accuracy of answers to all questions

**Datasets:**
- Image datasets: [MMVET](https://github.com/yuweihao/MM-Vet), [MMMU](https://mmmu-benchmark.github.io/), [MMBench](https://github.com/open-compass/MMBench), [OKVQA](https://okvqa.allenai.org/), [MathVista](https://mathvista.github.io/)
- Video datasets: [EgoSchema](https://egoschema.github.io/), and our own dataset for complex reasoning and video analysis

Refer section 5 and 6 in our [paper](https://arxiv.org/pdf/2405.18358) for complete set of results

## **What are the limitations of MMCT Agent? How can users minimize the impact of MMCT Agent's limitations when using the system?**

**Limitations:**
- MMCTAgent can still hallucinate and generate incorrect answers; additional measures are necessary to verify the reasoning chain
- **We do NOT recommend deploying MMCTAgent without content filtering** to ensure the safety of the results
- While MMCTAgent has shown promising results across various datasets, applying it to real-world scenarios requires further testing
- Dependency on external tools can introduce failure points if these tools fail or are unavailable
- The computational overhead of MMCTAgent may limit real-time applicability

## **What operational factors and settings allow for effective and responsible use of MMCT Agent?**

**Hardware Requirements:**
- GPU is necessary to support tools that are inferred locally
- Recommended VM specification: 1 x A100 80 GB, 64 cpu cores at 3.2GHz and 512 GB RAM

**Tools from Azure:**
- VIT (Vision Interpreter)
- OCR
- Object Detection
- Face Recognition
- Automatic Speech Recognition
- Access to GPT-4V

**Content Moderation & Safety:**
- Azure Content Moderation and Safety Shield to be enabled on both input to and output from MMCTAgent
