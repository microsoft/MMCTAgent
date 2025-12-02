# **MMCT - Image Pipeline**

[![](docs/multimedia/image-agent.png)](https://arxiv.org/pdf/2405.18358)

## **Overview**

Image Pipeline consists of Image Agent which is built on top of the **Multi-Modal Critical Thinking (MMCT)** ([arxiv.org/abs/2405.18358](https://arxiv.org/abs/2405.18358)) architecture, which leverages two collaborative agents:

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
from mmct.providers.azure import AzureLLMProvider
from mmct.config.providers import ImageAgentProviderConfig
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
import asyncio

# Configure credentials (or use api_key directly)
credentials = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

# Initialize the provider
provider = ImageAgentProviderConfig(
    llm_provider=AzureLLMProvider(
        endpoint="<your_endpoint>",
        deployment_name="<deployment_name>",
        model_name="<model_name>",
        api_version="<api_version>",
        credentials=credentials,  # Or use api_key="your-api-key"
    )
)

# Define the tools - refer to ImageQnaTools enum for available tools
tools = [ImageQnaTools.object_detection, ImageQnaTools.vit]

# Initialize the Image Agent
mmct_agent = ImageAgent(
    query="What objects are visible in this image?",
    image_path="path/to/your/image.jpg",
    tools=tools,
    use_critic_agent=True,  # Enable critic agent for improved accuracy
    stream=False,
    provider=provider
)

# Run the agent
response = asyncio.run(mmct_agent())
print(response.response)
```

### **Using Custom LLM Providers**

You can implement custom LLM providers for any vendor (Anthropic, Cohere, etc.) by inheriting from `BaseLLMProvider`:

```python
from mmct.providers.base import BaseLLMProvider
from mmct.config.providers import ImageAgentProviderConfig
from mmct.image_pipeline import ImageAgent, ImageQnaTools

# Your custom LLM provider implementation
class CustomLLMProvider(BaseLLMProvider):
    # Implement required methods: chat_completion() and get_autogen_client()
    pass

# Use your custom provider
custom_llm = CustomLLMProvider(api_key="your-api-key", model_name="your-model")
provider = ImageAgentProviderConfig(llm_provider=custom_llm)

mmct_agent = ImageAgent(
    query="Your query here",
    image_path="path/to/image.jpg",
    tools=[ImageQnaTools.vit],
    provider=provider
)
response = asyncio.run(mmct_agent())
```

For a complete working example of a custom Anthropic provider, see [`examples/image_agent.ipynb`](../../examples/image_agent.ipynb).

For detailed implementation instructions, refer to the [Providers Guide](../providers/README.md).

---