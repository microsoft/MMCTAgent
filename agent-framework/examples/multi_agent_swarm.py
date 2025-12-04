"""Example: Multi-agent swarm with handoffs.

This example demonstrates:
- Creating multiple specialized agents
- Setting up handoffs between agents
- Running a swarm with automatic routing
- Context transformation during handoffs
- Default debug logging for all hook events
"""

import asyncio

from credentials import AzureCredentials
from mmct_agent import Agent, Swarm, tool
from mmct_agent.llm import AzureOpenAIClient
from mmct_agent.config import Settings
from mmct_agent.core.swarm import SwarmConfig
from mmct_agent.observability import setup_logging, LogConfig


# === Research Tools ===

@tool(description="Search the web for information on a topic")
async def search_web(query: str) -> dict:
    """Search the web for information.
    
    Args:
        query: Search query string.
        
    Returns:
        Search results.
    """
    # Simulated search results
    return {
        "query": query,
        "results": [
            {
                "title": f"Introduction to {query}",
                "snippet": f"A comprehensive overview of {query} and its applications...",
                "url": f"https://example.com/{query.replace(' ', '-')}",
            },
            {
                "title": f"Latest developments in {query}",
                "snippet": f"Recent breakthroughs in {query} have shown promising results...",
                "url": f"https://research.example.com/{query.replace(' ', '-')}",
            },
        ],
    }


@tool(description="Read and summarize a document or article")
async def read_document(url: str) -> str:
    """Read a document from a URL.
    
    Args:
        url: URL of the document to read.
        
    Returns:
        Document content summary.
    """
    # Simulated document reading
    return f"""
Document from {url}:

This is a comprehensive article covering the topic in depth.
Key points include:
1. Historical background and context
2. Current state of the art
3. Future directions and challenges
4. Practical applications

The research shows promising results in multiple areas.
"""


# === Writing Tools ===

@tool(description="Check grammar and style of text")
def check_grammar(text: str) -> dict:
    """Check grammar and style.
    
    Args:
        text: Text to check.
        
    Returns:
        Grammar check results.
    """
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "issues_found": 0,
        "suggestions": ["Consider adding more specific examples"],
    }


@tool(description="Generate an outline for a topic")
async def generate_outline(topic: str, sections: int = 5) -> list:
    """Generate a document outline.
    
    Args:
        topic: Topic for the outline.
        sections: Number of sections.
        
    Returns:
        List of section headings.
    """
    return [
        f"1. Introduction to {topic}",
        f"2. Key Concepts and Terminology",
        f"3. Current Applications",
        f"4. Challenges and Limitations",
        f"5. Future Outlook",
    ][:sections]


async def main():
    # Set up logging with DEBUG level for detailed execution logs
    setup_logging(LogConfig(level="DEBUG", format="text"))
    
    # Load settings
    settings = Settings()
    
    # Create LLM client
    llm_client = AzureOpenAIClient(
        endpoint=settings.azure_openai_endpoint,
        deployment=settings.azure_openai_deployment,
        api_version=settings.azure_openai_api_version,
        azure_credential=AzureCredentials.get_credentials()
    )
    
    # Create specialized agents
    researcher = Agent(
        name="researcher",
        system_prompt="""You are a research specialist. Your job is to:
1. Search for information on the given topic
2. Read and analyze relevant documents
3. Compile key findings""",
        llm_client=llm_client,
        tools=[search_web, read_document],
    )
    
    writer = Agent(
        name="writer",
        system_prompt="""You are a skilled content writer. Your job is to:
1. Take research findings and create well-structured content
2. Generate outlines when needed
3. Write clear, engaging prose
""",
        llm_client=llm_client,
        tools=[generate_outline],
    )
    
    editor = Agent(
        name="editor",
        system_prompt="""You are a meticulous editor. Your job is to:
1. Review the content for clarity and accuracy
2. Check grammar and style
3. Provide the final polished version

You provide the final output - do not hand off to anyone else.""",
        llm_client=llm_client,
        tools=[check_grammar],
    )
    
    # Set up handoffs with content schemas
    researcher.register_handoff(
        target="writer",
        description="Hand off to the writer when research is complete. Provide the research findings.",
        content_schema={
            "type": "object",
            "properties": {
                "research_findings": {
                    "type": "string",
                    "description": "Comprehensive research findings including key facts, data points, and insights gathered.",
                },
                "sources": {
                    "type": "string",
                    "description": "List of sources consulted during research.",
                },
            },
            "required": ["research_findings"],
        },
    )
    
    writer.register_handoff(
        target="editor",
        description="Hand off to the editor when the draft is complete. Provide the full article draft.",
        content_schema={
            "type": "object",
            "properties": {
                "draft": {
                    "type": "string",
                    "description": "The complete article draft ready for editing.",
                },
                "notes": {
                    "type": "string",
                    "description": "Any notes or areas of concern for the editor.",
                },
            },
            "required": ["draft"],
        },
    )
    
    # Create swarm (default debug logging hooks are used automatically)
    swarm = Swarm(
        agents=[researcher, writer, editor],
        config=SwarmConfig(
            max_iterations=10,
        ),
    )
    
    print("=" * 60)
    print("MMCT Agent Framework - Multi-Agent Swarm Example")
    print("=" * 60)
    print("\n📋 Task: Research and write a short article")
    print("-" * 60)
    
    # Run the swarm
    result = await swarm.run(
        initial_agent="researcher",
        task="Research and write a short article about the impact of AI on software development.",
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📄 FINAL OUTPUT")
    print("=" * 60)
    
    if result.final_response:
        print(result.final_response.content)
    
    print("\n" + "=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Agents used: {' → '.join(result.agents_used)}")
    print(f"Total iterations: {result.iterations}")
    print(f"Total tokens: {result.total_token_usage.total_tokens}")
    print(f"Total latency: {result.total_latency_ms:.2f}ms")
    print(f"Success: {result.success}")
    
    # Explicitly save memory for debugging (optional)
    # saved_path = await swarm.save_memory(path="./debug_logs")
    # print(f"Memory saved to: {saved_path}")


if __name__ == "__main__":
    asyncio.run(main())
