# **MMCT - Image Pipeline**
[![](/docs/multimedia/imageAgent.webp)](https://arxiv.org/pdf/2405.18358)
## **Overview**

Image Pipeline consists of Image Agent which is is built on top of the **Multi-Modal Critical Thinking (MMCT)** ([arxiv.org/abs/2405.18358](https://arxiv.org/abs/2405.18358)) architecture, which leverages two collaborative agents:

- **Planner**: Generates an initial response based on the provided input. It uses a set of default tools from `ImageQnaTools` but can be customized.
- **Critic (optional)**: Evaluates the planner’s response and provides feedback for improvement. This feedback loop helps increase accuracy and quality.

By default, the critic agent is enabled. Users can disable it by setting `use_critic_agent=False` during initialization.

> **Note:** Disabling the critic agent skips the feedback loop and may reduce the accuracy of the final response.

---

## **Tool Configuration**

The planner supports the following tools via the `ImageQnaTools` enum:

- `ImageQnaTools.object_detection` – This tool detects the object in the image.
- `ImageQnaTools.ocr` – for extracting text content.
- `ImageQnaTools.recog` – This tool recognise the objects in the image.
- `ImageQnaTools.vit` – for high-level visual understanding using vision transformers.

Users can pass a list of tools via the `tools` parameter to override the defaults.
---

## **Tool Workflow**

1. **Input Processing**

   - The user provides an image input along with a query.
   - The system is initialized with a set of tools (default or user-defined) and an optional critic agent.

2. **Planner Agent Execution**

   - The **Planner** is the core agent that first analyzes the input.
   - It selects appropriate tools from the `ImageQnaTools` enum based on the task:
     - `object_detection`: Detects objects in the image.
     - `ocr`: Extracts textual information.
     - `recog`: Recognizes objects/entities.
     - `vit`: Performs high-level visual reasoning using vision transformers.
   - The planner generates an initial response based on these tools.

3. **Critic Agent Feedback (Optional)**

   - If `use_critic_agent=True` (default), the **Critic** reviews the planner’s output.
   - It evaluates the quality and correctness of the response.
   - If needed, it provides feedback, prompting the planner to revise its output.
   - This loop can iterate to refine the final result.

4. **Final Response**
   - The system returns a response that integrates insights from selected tools and (optionally) the critic's feedback.
   - If the critic is disabled, the planner's output is returned directly, which may be less refined.

> **Note:** Disabling the critic speeds up processing but may affect the accuracy and depth of the response.

---

## **Usage**

Below is the script to get started with the MMCT Image Agent. 

> MMCT Image Agent

```python
from mmct.image_pipeline import ImageAgent, ImageQnaTools
import asyncio
import ast

# user query
query = ""
# define the tools, you can refer to the Enum definition of Tools to get the idea for available tools
tools = [ImageQnaTools.object_detection, ImageQnaTools.vit]
# flag variable whether you want to initialize Critic Agent or not.
use_critic_agent = True
# flag variable whether you have to stream or not.
stream = False
# initialize the Image Agent.
mmct_agent = ImageAgent(
    query=query,
    image_path=image_path,
    tools=tools,
    use_critic_agent=use_critic_agent,
    stream=stream,
)
response = asyncio.run(mmct_agent())
print(response.response)
```
---