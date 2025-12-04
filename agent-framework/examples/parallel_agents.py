"""Example: Parallel agent execution.

This example demonstrates:
- Running multiple agents in parallel on the same task
- Aggregating results from multiple agents
- Comparing different agent responses
"""

import asyncio

from credentials import AzureCredentials
from mmct_agent import Agent, Swarm, tool
from mmct_agent.llm import AzureOpenAIClient
from mmct_agent.config import Settings
from mmct_agent.observability import setup_logging, LogConfig


@tool(description="Analyze sentiment of text")
def analyze_sentiment(text: str) -> dict:
    """Analyze text sentiment.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Sentiment analysis results.
    """
    # Simulated sentiment analysis
    return {
        "sentiment": "positive",
        "confidence": 0.85,
        "keywords": ["innovative", "efficient", "promising"],
    }


@tool(description="Extract key entities from text")
def extract_entities(text: str) -> list:
    """Extract named entities from text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        List of extracted entities.
    """
    # Simulated entity extraction
    return [
        {"type": "TECHNOLOGY", "value": "AI"},
        {"type": "CONCEPT", "value": "automation"},
        {"type": "FIELD", "value": "software development"},
    ]


@tool(description="Summarize text in bullet points")
def summarize_bullets(text: str, num_bullets: int = 3) -> list:
    """Summarize text as bullet points.
    
    Args:
        text: Text to summarize.
        num_bullets: Number of bullet points.
        
    Returns:
        List of summary bullet points.
    """
    # Simulated summarization
    return [
        "Key point 1 from the text",
        "Key point 2 from the text",
        "Key point 3 from the text",
    ][:num_bullets]


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
    
    # Create specialized analyzers
    sentiment_analyst = Agent(
        name="sentiment_analyst",
        system_prompt="""You are a sentiment analysis expert. 
Analyze the emotional tone and sentiment of the given text.
Use the sentiment analysis tool and provide insights.""",
        llm_client=llm_client,
        tools=[analyze_sentiment],
    )
    
    entity_extractor = Agent(
        name="entity_extractor",
        system_prompt="""You are an entity extraction specialist.
Identify and extract key entities (people, organizations, technologies, concepts) from text.
Use the entity extraction tool and explain the significance of each entity.""",
        llm_client=llm_client,
        tools=[extract_entities],
    )
    
    summarizer = Agent(
        name="summarizer",
        system_prompt="""You are a summarization expert.
Create concise, informative summaries of text content.
Use the bullet point summarization tool and add context.""",
        llm_client=llm_client,
        tools=[summarize_bullets],
    )
    
    # Create swarm
    swarm = Swarm(agents=[sentiment_analyst, entity_extractor, summarizer])
    
    print("=" * 70)
    print("MMCT Agent Framework - Parallel Agent Execution")
    print("=" * 70)
    
    # Sample text to analyze
    sample_text = """
    Artificial Intelligence is revolutionizing the software development industry.
    New tools powered by machine learning are helping developers write better code
    faster than ever before. Companies like Microsoft, Google, and OpenAI are
    leading the charge with innovations in code completion, automated testing,
    and intelligent debugging. While some worry about job displacement, many
    experts believe AI will augment rather than replace human developers,
    allowing them to focus on higher-level design and creative problem-solving.
    """
    
    print("\n📝 Input Text:")
    print("-" * 70)
    print(sample_text.strip())
    print("-" * 70)
    
    print("\n🔄 Running 3 agents in parallel...")
    print("-" * 70)
    
    # Run agents in parallel
    task = f"Analyze the following text:\n\n{sample_text}"
    
    responses = await swarm.run_parallel(
        agents=["sentiment_analyst", "entity_extractor", "summarizer"],
        task=task,
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 PARALLEL ANALYSIS RESULTS")
    print("=" * 70)
    
    for response in responses:
        print(f"\n🤖 {response.agent_name.upper()}")
        print("-" * 50)
        print(response.content)
        print(f"\n  ⏱️  Latency: {response.latency_ms:.2f}ms")
        print(f"  🔧 Tools used: {[r.name for r in response.tool_results]}")
        print(f"  📊 Tokens: {response.token_usage.total_tokens}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 EXECUTION SUMMARY")
    print("=" * 70)
    
    total_tokens = sum(r.token_usage.total_tokens for r in responses)
    max_latency = max(r.latency_ms for r in responses)
    
    print(f"Total tokens used: {total_tokens}")
    print(f"Max latency (parallel): {max_latency:.2f}ms")
    print(f"Agents executed: {len(responses)}")
    print("All agents ran simultaneously - faster than sequential execution!")


if __name__ == "__main__":
    asyncio.run(main())
