"""Example: Basic agent with tools.

This example demonstrates:
- Creating an agent with tools
- Running the agent with user input
- Streaming responses
- Tracking token usage
"""

import asyncio

from credentials import AzureCredentials
from mmct_agent import Agent, tool
from mmct_agent.llm import AzureOpenAIClient
from mmct_agent.config import Settings
from mmct_agent.observability import setup_logging, LogConfig


# Define tools using the @tool decorator
@tool(description="Add two numbers together")
async def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: First number to add.
        b: Second number to add.
        
    Returns:
        Sum of the two numbers.
    """
    return a + b


@tool(description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    
    Args:
        a: First number.
        b: Second number.
        
    Returns:
        Product of the two numbers.
    """
    return a * b


@tool(description="Get the current weather for a city")
async def get_weather(city: str) -> dict:
    """Get weather information for a city.
    
    Args:
        city: Name of the city.
        
    Returns:
        Weather information dictionary.
    """
    # Simulated weather data
    return {
        "city": city,
        "temperature": 72,
        "conditions": "sunny",
        "humidity": 45,
    }


async def main():
    # Set up logging
    setup_logging(LogConfig(level="INFO", format="text"))
    
    # Load settings
    settings = Settings()
    
    # Create LLM client
    llm_client = AzureOpenAIClient(
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        azure_credential=AzureCredentials.get_credentials(),
    )
    
    # Create agent with tools
    agent = Agent(
        name="assistant",
        system_prompt="""You are a helpful assistant with access to math and weather tools.
Use the tools when needed to help answer questions accurately.
Always explain your reasoning and the results you get from tools.""",
        llm_client=llm_client,
        tools=[add, multiply, get_weather],
    )
    
    print("=" * 60)
    print("MMCT Agent Framework - Basic Agent Example")
    print("=" * 60)
    
    # Example 1: Simple tool use
    print("\n📝 Example 1: Math calculation")
    print("-" * 40)
    
    response = await agent.run("What is 15 + 27, and then multiply that result by 3?")
    print(f"Response: {response.content}")
    print(f"Tools used: {[r.name for r in response.tool_results]}")
    print(f"Token usage: {response.token_usage.total_tokens} tokens")
    
    # Reset agent memory for next example
    agent.reset()
    
    # Example 2: Multiple tool calls
    print("\n📝 Example 2: Weather query")
    print("-" * 40)
    
    response = await agent.run("What's the weather like in San Francisco?")
    print(f"Response: {response.content}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    
    # Example 3: Streaming response
    print("\n📝 Example 3: Streaming response")
    print("-" * 40)
    print("Streaming: ", end="", flush=True)
    
    agent.reset()
    async for chunk in agent.run_stream("Write a haiku about programming in 500 words"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")
    
    # Print total token usage
    print("=" * 60)
    print(f"Total tokens used: {agent.total_token_usage.total_tokens}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
