"""Example: Memory strategies demonstration.

This example demonstrates:
- Different memory strategies
- How they handle long conversations
- Token usage comparison
- Memory persistence
"""

import asyncio

from credentials import AzureCredentials
from mmct_agent import Agent, tool
from mmct_agent.llm import AzureOpenAIClient
from mmct_agent.config import Settings
from mmct_agent.memory import (
    SlidingWindowMemory,
    TokenBasedMemory,
    SummarizationMemory,
    AdaptiveMemory,
)
from mmct_agent.observability import setup_logging, LogConfig


@tool(description="Store a fact for later retrieval")
def store_fact(fact: str) -> str:
    """Store a fact.
    
    Args:
        fact: The fact to store.
        
    Returns:
        Confirmation message.
    """
    return f"Stored: {fact}"


async def run_conversation(agent: Agent, messages: list[str]) -> None:
    """Run a series of messages through an agent.
    
    Args:
        agent: Agent to use.
        messages: List of user messages.
    """
    for i, msg in enumerate(messages, 1):
        print(f"  [{i}] User: {msg[:50]}...")
        response = await agent.run(msg)
        print(f"      Agent: {response.content[:80] if response.content else 'No response'}...")
        
    # Show memory stats
    token_count = await agent.memory.get_token_count()
    raw_count = len(agent.memory.get_raw_messages())
    context_messages = await agent.memory.get_messages()
    
    print(f"\n  📊 Memory Stats:")
    print(f"     Strategy: {agent.memory.strategy_name}")
    print(f"     Raw messages: {raw_count}")
    print(f"     Context messages: {len(context_messages)}")
    print(f"     Approximate tokens: {token_count}")


async def main():
    # Set up logging
    setup_logging(LogConfig(level="WARNING", format="text"))
    
    # Load settings
    settings = Settings()
    
    # Create LLM client
    llm_client = AzureOpenAIClient(
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        azure_credential=AzureCredentials.get_credentials(),
    )
    
    # Test messages that would normally grow context quickly
    test_messages = [
        "Hi! I'm learning about Python programming. Can you help?",
        "What are the main data structures in Python?",
        "Tell me more about lists and how they work.",
        "How do dictionaries differ from lists?",
        "What about sets? When should I use them?",
        "Can you explain list comprehensions with examples?",
        "What are generators and how do they differ from lists?",
        "How does Python handle memory management?",
        "What is garbage collection in Python?",
        "Tell me about decorators and their use cases.",
        "How do context managers work?",
        "What's the difference between args and kwargs?",
        "Explain Python's class system briefly.",
        "What are metaclasses?",
        "Finally, summarize the key Python concepts we discussed.",
    ]
    
    print("=" * 70)
    print("MMCT Agent Framework - Memory Strategies Demonstration")
    print("=" * 70)
    
    # 1. Sliding Window Memory
    print("\n📚 Strategy 1: Sliding Window (keeps last 5 messages)")
    print("-" * 70)
    
    agent1 = Agent(
        name="sliding_window_agent",
        system_prompt="You are a helpful Python tutor. Keep responses brief.",
        llm_client=llm_client,
        tools=[store_fact],
        memory=SlidingWindowMemory(window_size=5),
    )
    
    await run_conversation(agent1, test_messages[:7])
    
    # 2. Token-Based Memory
    print("\n📚 Strategy 2: Token-Based (max 2000 tokens)")
    print("-" * 70)
    
    agent2 = Agent(
        name="token_based_agent",
        system_prompt="You are a helpful Python tutor. Keep responses brief.",
        llm_client=llm_client,
        tools=[store_fact],
        memory=TokenBasedMemory(max_tokens=2000, token_buffer=500),
    )
    
    await run_conversation(agent2, test_messages[:7])
    
    # 3. Summarization Memory
    print("\n📚 Strategy 3: Summarization (summarizes older messages)")
    print("-" * 70)
    
    agent3 = Agent(
        name="summarization_agent",
        system_prompt="You are a helpful Python tutor. Keep responses brief.",
        llm_client=llm_client,
        tools=[store_fact],
        memory=SummarizationMemory(
            llm_client=llm_client,
            summarization_threshold=1500,
            summary_max_tokens=300,
        ),
    )
    
    await run_conversation(agent3, test_messages[:7])
    
    # 4. Adaptive Memory
    print("\n📚 Strategy 4: Adaptive (automatically selects best strategy)")
    print("-" * 70)
    
    agent4 = Agent(
        name="adaptive_agent",
        system_prompt="You are a helpful Python tutor. Keep responses brief.",
        llm_client=llm_client,
        tools=[store_fact],
        memory=AdaptiveMemory(
            llm_client=llm_client,
            max_tokens=2000,
            window_size=10,
            summarization_threshold=1500,
        ),
    )
    
    await run_conversation(agent4, test_messages[:7])
    
    # Memory persistence example - now explicit via agent.save_memory()
    print("\\n📚 Memory Persistence Example")
    print("-" * 70)
    
    # Save agent4's memory explicitly
    saved_path = await agent4.save_memory(
        path="./memory_demo_logs",
        session_id="demo_session",
    )
    print(f"  ✅ Saved memory to {saved_path}")
    
    # Show operation log
    print("\n  📜 Memory Operation Log (last 5 operations):")
    for op in agent4.memory.get_operation_log()[-5:]:
        print(f"     - {op['operation']}: {op.get('message_role', op.get('count', ''))}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
