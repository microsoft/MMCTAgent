"""
Query Federator Agent for intelligent query routing in video analysis.

This module implements a query federator that:
1. Classifies queries as simple (keyword/vague) or complex (requiring deep analysis)
2. Routes simple queries directly to get_video_analysis() or get_context() tools
3. Routes complex queries to the planner-critic team for comprehensive analysis
"""

import asyncio
import json
import os
import re
import logging
from typing import Optional, Dict, Any
from loguru import logger
from dotenv import load_dotenv

# Suppress autogen internal logging
logging.getLogger("autogen").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

from typing import Annotated
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult

from mmct.video_pipeline.core.tools.get_context import get_context
from mmct.video_pipeline.core.tools.get_video_summary import get_video_summary
from mmct.video_pipeline.core.tools.get_object_collection import get_object_collection
from mmct.video_pipeline.core.tools.video_qna import video_qna
from mmct.video_pipeline.core.tools.cache import Cache
from mmct.providers.factory import provider_factory

from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache as DiskCache

load_dotenv(override=True)


# Prompts for the classifier and simple query handler
QUERY_CLASSIFIER_SYSTEM_PROMPT = """
You are a query classification expert for video analysis systems.

Your job is to analyze user queries and classify them into two categories:

1. **SIMPLE**: Queries that are:
    - Vague or general (e.g., "tell me about the video", "what's in this video")
    - Keyword-based searches (e.g., "car", "person in blue", "basketball")
    - Asking for basic summaries or overviews
    - Looking for specific objects/things without complex reasoning

2. **COMPLEX**: Queries that require:
    - Deep temporal analysis (e.g., "what happened after the person left")
    - Multi-step reasoning (e.g., "why did X happen and how does it relate to Y")
    - Specific frame-level visual analysis
    - Cross-referencing multiple video segments
    - Detailed explanations requiring transcript + visual correlation
    - Cause-and-effect relationships
    - Comparative analysis

Available Tools:
- **get_video_summary**: Retrieves high-level video summaries. Can be used for video discovery (without video_id/URL) or to get summary of a specific video (with video_id/URL)
- **get_object_collection**: Retrieves object collection data including object descriptions, counts, and first_seen timestamps. REQUIRES video_id/URL (call get_video_summary first if not available)
- **get_context**: Retrieves sections from videos which are most relevant to the query (targeted retrieval)
- **planner_team**: Complex multi-step reasoning and analysis

Respond ONLY with a JSON object:
{
     "classification": "SIMPLE" or "COMPLEX",
     "reasoning": "Brief explanation of why this query falls into this category",
     "recommended_tool": "get_video_summary" or "get_object_collection" or "get_context" or "planner_team"
}

Guidelines:
- If uncertain, classify as SIMPLE (we can always escalate if needed)
- For general summaries or overviews, prefer "get_video_summary"
- For object counting, tracking, or appearance details, prefer "get_object_collection"
- For keyword-based searches or querying specific details, prefer "get_context"
- For complex reasoning, use "planner_team"
"""

SIMPLE_QUERY_HANDLER_SYSTEM_PROMPT = """
You handle straightforward video-analysis questions.

Tools (pick exactly one, call it once, then answer):
- get_video_summary: For video discovery or high-level summaries.
- get_object_collection: For object counts, tracking, or appearance details (needs video_id; call get_video_summary first if missing).
- get_context: For transcript or targeted detail searches.

Workflow:
1. Run the single most relevant tool.
2. Use its output to craft a concise, markdown-formatted answer.
3. If the tool returns nothing relevant, say "Not enough information in context."

Response format (JSON only, followed by TERMINATE):
{
    "answer": "<Markdown answer or 'Not enough information in context'>",
    "source": ["TEXTUAL", "VISUAL"],
    "videos": [
        {
            "hash_id": "<hash_video_id>",
            "url": "<video_url>",
            "timestamps": [["HH:MM:SS", "HH:MM:SS"]]
        }
    ]
}
TERMINATE

Formatting rules:
- Keep the answer self-contained; no references like "the video shows."
- Do not include timestamps inside the answer text.
- Use markdown for clarity (lists, headings, emphasis when useful).
- Only list videos/timestamps actually used.
- "source" should reflect whether information came from transcript (TEXTUAL), visuals (VISUAL), or both.
"""


class QueryFederator:
    """
    Query Federator for intelligent routing of video analysis queries.

    This class implements a two-stage approach:
    1. Classification: Determines if query is simple or complex
    2. Routing: Sends simple queries to direct tools, complex queries to planner-critic team

    All responses follow a standardized JSON structure:
    {
        "result": {
            "answer": "<Markdown-formatted answer>",
            "source": ["TEXTUAL", "VISUAL"],
            "videos": [{"hash_id": "...", "url": "...", "timestamps": [...]}]
        },
        "tokens": {"total_input": int, "total_output": int}
    }

    Args:
        query (str): The natural language question to be answered
        video_id (str, optional): The unique identifier of the video
        url (str, optional): The video URL
        use_critic_agent (bool): Whether to use critic agent for complex queries
        index_name (str): Vector index name for context retrieval
        llm_provider (object, optional): LLM provider instance
        cache (bool): Whether to enable caching for model responses
    """
    
    DEFAULT_OUTPUT = {
        "answer": "Not enough information in context",
        "source": [],
        "videos": []
    }

    def __init__(
        self,
        query: str,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        use_critic_agent: bool = True,
        index_name: str = None,
        llm_provider: Optional[object] = None,
        cache: Optional[bool] = False,
        semantic_cache: Optional[bool] = True
    ):
        self.query = query
        self.video_id = video_id
        self.url = url
        self.use_critic_agent = use_critic_agent
        self.index_name = index_name
        self.cache = cache
        self.llm_provider = llm_provider
        self.semantic_cache = semantic_cache

        # Initialize providers if not provided
        if self.llm_provider is None:
            self.llm_provider = provider_factory.create_llm_provider()

        self.model_client = self.llm_provider.get_autogen_client()
        self.model_client_no_parallel_tool_calls = self.llm_provider.get_autogen_client_for_no_tools_agent()

        # Initialize cache instance
        self.cache_instance = Cache()

        self.classifier_agent = None
        self.simple_handler_agent = None
        self.classification_result = None

    async def _classify_query(self) -> Dict[str, Any]:
        """
        Classify the query as SIMPLE or COMPLEX using an LLM-based classifier.

        Returns:
            Dict containing classification, reasoning, and recommended_tool
        """
        logger.info(f"Classifying query: {self.query}")

        # Create classifier agent
        classifier = AssistantAgent(
            name="query_classifier",
            model_client=self.model_client_no_parallel_tool_calls,
            model_client_stream=False,
            description="Classifies queries as SIMPLE or COMPLEX for video analysis",
            system_message=QUERY_CLASSIFIER_SYSTEM_PROMPT,
        )
        
        # Run classification
        classification_task = f"Classify this query: {self.query}\n\nRespond with ONLY the JSON classification."
        result = await Console(classifier.run_stream(task=classification_task))

        # Extract the classification from the last message
        last_message = result.messages[-1].content if result.messages else "{}"
                
        # Parse the JSON response
        import json
        try:
            # Try to find JSON in the response
            start_idx = last_message.find("{")
            end_idx = last_message.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = last_message[start_idx:end_idx]
                classification_result = json.loads(json_str)
            else:
                # Default to SIMPLE if parsing fails
                logger.warning("Failed to parse classification JSON, defaulting to SIMPLE")
                classification_result = {
                    "classification": "SIMPLE",
                    "reasoning": "Failed to parse, defaulting to SIMPLE",
                    "recommended_tool": "get_video_analysis"
                }
        except json.JSONDecodeError:
            logger.warning("JSON decode error, defaulting to SIMPLE")
            classification_result = {
                "classification": "SIMPLE",
                "reasoning": "JSON decode error, defaulting to SIMPLE",
                "recommended_tool": "get_video_summary"
            }

        logger.info(f"Classification result: {classification_result}")
        return classification_result

    async def _handle_simple_query(self, recommended_tool: str) -> Dict[str, Any]:
        """
        Handle simple queries using direct tool calls.

        Creates an assistant agent with access to get_video_summary (for high-level summaries 
        and video discovery), get_object_collection (for object tracking and counting), and 
        get_context (for targeted content retrieval) tools, executes the query, and parses 
        the response into a standardized JSON structure.

        Args:
            recommended_tool: The tool recommended by the classifier
                            ("get_video_summary" or "get_object_collection" or "get_context")

        Returns:
            Dict containing:
            - result: Parsed JSON dict with "answer", "source", and "videos" keys
            - tokens: Token usage dict with "total_input" and "total_output"
        """
        logger.info(f"Handling simple query with tool: {recommended_tool}")

        # Create simple handler agent with access to both tools
        simple_handler = AssistantAgent(
            name="simple_query_handler",
            model_client=self.model_client,
            system_message=SIMPLE_QUERY_HANDLER_SYSTEM_PROMPT,
            tools=[get_video_summary, get_object_collection, get_context],
            reflect_on_tool_use=True,
            max_tool_iterations=5,  # Limited iterations for simple queries
        )

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(2)
        team = RoundRobinGroupChat(
            participants=[simple_handler],
            termination_condition=termination,
        )

        # Construct task with guidance on which tool to prefer
        task = (
            f"Query: {self.query}\n\n"
            f"Instructions:\n"
            + (f"- Video ID: {self.video_id}\n" if self.video_id else "")
            + (f"- URL: {self.url}\n" if self.url else "")
            + f"- Index name: {self.index_name}\n"
            + f"- Recommended tool: {recommended_tool}\n\n"
            + "Please answer this query using the appropriate tool(s)."
        )

        result = await Console(team.run_stream(task=task))
        tokens = await self._calculate_total_tokens(result.messages)
        
        # Extract and parse JSON from the response
        last_message = result.messages[-1].content if result.messages else "{}"
    
        try:
            # Remove TERMINATE keyword
            content = last_message.replace('TERMINATE', '').strip()
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
            
            parsed_result = json.loads(json_str)
            return {"result": parsed_result, "tokens": tokens}
        
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                    "result": {
                        "answer": "Error parsing response",
                        "source": [],
                        "videos": []
                    }, 
                    "tokens": tokens
            }
            
    async def _handle_complex_query(self) -> Dict[str, Any]:
        """
        Handle complex queries by delegating to the planner-critic team.

        Calls the video_qna function with stream=True to utilize the full
        planner-critic workflow for multi-step reasoning and analysis.

        Returns:
            Dict containing:
            - result: Parsed JSON dict with "answer", "source", and "videos" keys
            - tokens: Token usage dict with "total_input" and "total_output"
        """
        logger.info("Handling complex query with planner-critic team")

        # Use the existing VideoQnA class for complex queries
        result = await video_qna(
            query=self.query,
            video_id=self.video_id,
            url=self.url,
            use_critic_agent=self.use_critic_agent,
            index_name=self.index_name,
            stream=True,
            llm_provider=self.llm_provider,
            cache=self.cache,
        )
        return result

    async def _calculate_total_tokens(self, messages) -> dict:
        """
        Calculate total input and output tokens from messages.

        Args:
            messages: List of message objects from TaskResult

        Returns:
            dict: {'total_input': int, 'total_output': int}
        """
        total_input = 0
        total_output = 0

        for message in messages:
            usage = getattr(message, "models_usage", None)
            if usage:
                total_input += getattr(usage, "prompt_tokens", 0) or 0
                total_output += getattr(usage, "completion_tokens", 0) or 0

        return {"total_input": total_input, "total_output": total_output}

    async def run(self) -> Dict[str, Any]:
        """
        Main execution method for the query federator.

        Workflow:
        1. Classifies the query as SIMPLE or COMPLEX
        2. Routes to appropriate handler (simple_handler or planner_team)
        3. Returns standardized response

        Returns:
            Dict containing:
            - result: Parsed JSON dict with "answer", "source", and "videos"
            - tokens: Token usage dict with "total_input" and "total_output"
        """
        # Step 0: Check Cache (Exact Match on question)
        # self.query
        if self.semantic_cache and self.cache_instance:
            tokens = {"total_input": 0, "total_output": 0}
            try:
                cache_response = await self.cache_instance.get_cache_response(question = self.query, url = self.url)
                if isinstance(cache_response['videos'], str):
                    cache_response['videos'] = json.loads(cache_response['videos'])
                return {
                        "result": {
                            "answer": cache_response['answer'],
                            "source": cache_response['source'],
                            "videos": cache_response['videos']
                        },
                        "tokens": tokens
                }
            except:
                return {
                    "result": {
                        "answer": "Error fetching cache response",
                        "source": [],
                        "videos": []
                    },
                    "tokens": tokens
                }

        # Step 1: Classify the query
        classification = await self._classify_query()
        self.classification_result = classification

        # Step 2: Route based on classification
        if classification["classification"] == "SIMPLE":
            logger.info("Query classified as SIMPLE - using direct tools")
            result = await self._handle_simple_query(classification["recommended_tool"])
        else:
            logger.info("Query classified as COMPLEX - using planner-critic team")
            result = await self._handle_complex_query()

        try:
            result_data = result.get("result", {})
            answer = result_data.get("answer", "")
            source = result_data.get("source", [])
            videos = result_data.get("videos", [])

            # Only cache if we have a valid answer
            if answer and answer != "Not enough information in context":
                await self.cache_instance.set_cache(
                    question=self.query,
                    answer=answer,
                    source=source,
                    videos=videos,
                    url=self.url
                )
        except Exception as e:
            logger.warning(f"Failed to set cache: {e}")

        return result


async def query_federator(
    query: Annotated[str, "The question to be answered based on the content of the video."],
    video_id: Annotated[Optional[str], "The unique identifier of the video."] = None,
    url: Annotated[Optional[str], "The URL of the video to filter out the search results"] = None,
    use_critic_agent: Annotated[
        bool, "Set to True to enable a critic agent for complex queries."
    ] = True,
    index_name: Annotated[
        str, "Vector index name for context retrieval"
    ] = "education-video-index-v2",
    llm_provider: Optional[object] = None,
    cache: Annotated[bool, "Set to True to enable cache for model responses."] = True,
    semantic_cache: Annotated[bool, "Set to True to enable semantic cache powered by search index"] = True
) -> Dict[str, Any]:
    """
    Query Federator for intelligent video query routing.

    This function implements a smart routing system that:
    1. Classifies queries as SIMPLE (vague/keyword-based) or COMPLEX (requiring deep analysis)
    2. Routes SIMPLE queries directly to get_video_summary(), get_object_collection(), or get_context() tools
    3. Routes COMPLEX queries to the planner-critic team for comprehensive analysis

    SIMPLE queries include:
    - Vague questions ("tell me about this video")
    - Keyword searches ("car", "person in blue")
    - Basic summaries or object counting
    - Object tracking and appearance details

    COMPLEX queries include:
    - Temporal analysis ("what happened after...")
    - Multi-step reasoning
    - Frame-level visual analysis
    - Comparative or causal relationships

    Args:
        query: The natural language question to be answered
        video_id: The unique identifier of the video (optional)
        url: The URL of the video (optional)
        use_critic_agent: Whether to use critic agent for complex queries
        index_name: Vector index name for context retrieval
        llm_provider: LLM provider instance (optional)
        cache: Whether to enable caching for model responses

    Returns:
        Dict containing:
        - result: Dict with "answer" (markdown text), "source" (list), and "videos" (list)
        - tokens: Dict with "total_input" and "total_output" token counts
        
        Example response:
        {
            "result": {
                "answer": "The video shows...",
                "source": ["VISUAL", "TEXTUAL"],
                "videos": [{"hash_id": "...", "url": "...", "timestamps": [["00:00:00", "00:01:30"]]}]
            },
            "tokens": {"total_input": 1234, "total_output": 567}
        }
    """
    federator = QueryFederator(
        query=query,
        video_id=video_id,
        url=url,
        use_critic_agent=use_critic_agent,
        index_name=index_name,
        llm_provider=llm_provider,
        cache=cache,
        semantic_cache=semantic_cache
    )

    return await federator.run()


if __name__ == "__main__":
    # Example usage
    test_queries = [
        # Simple queries
        "Can you explain the procedure for creating Jeevamrit (Jeevamrit)?",
    ]

    async def test_federator():
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: {query}")
            print('='*60)

            result = await query_federator(
                query=query,
                index_name="dg-jharkhand-kv",
                use_critic_agent=False,
                cache=False,
                semantic_cache=False
            )
            print(f"Result: {result}")
    
    # Uncomment to run tests
    asyncio.run(test_federator())
