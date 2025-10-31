# Standard Library
import asyncio
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Local Imports
from mmct.video_pipeline.core.tools.video_qna import video_qna
from mmct.video_pipeline.prompts_and_description import (
    VIDEO_AGENT_SYSTEM_PROMPT,
    VideoAgentResponse,
)
from autogen_agentchat.ui import Console
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory

# Load environment variables
load_dotenv(override=True)


class VideoAgent:
    """
    Simplified agent for video question answering using the updated video_qna function with Swarm orchestration.

    This agent provides a clean interface that:
    1. Calls video_qna (v2 with Swarm) with the provided parameters
    2. Formats the response using LLM with structured output
    3. Returns a properly structured VideoAgentResponse

    Args:
        query (str): The natural language question about video content.
        index_name (str): Name of the Azure Cognitive Search index for video retrieval.
        video_id (Optional[str]): Specific video ID to query. Defaults to None.
        url (Optional[str]): URL to filter the search results for that particular video. Defaults to None.
        use_critic_agent (bool): Whether to use the critic agent for validation. Defaults to True.
        stream (bool): Whether to stream the response output. Defaults to False.
        llm_provider (Optional[object]): LLM provider instance. Defaults to None (uses config).

    Example:
        Basic usage with query and index:
        ```python
        video_agent = VideoAgent(
            query="What are the benefits of organic farming?",
            index_name="farming-video-index"
        )
        result = await video_agent()
        print(result.response)
        ```

        With specific video ID:
        ```python
        video_agent = VideoAgent(
            query="Explain the farming technique shown",
            index_name="farming-video-index",
            video_id="abc123def456"
        )
        result = await video_agent()
        ```

        With URL and streaming:
        ```python
        video_agent = VideoAgent(
            query="Summarize this farming video",
            index_name="farming-video-index",
            url="https://video-url.mp4",
            stream=True
        )
        result = await video_agent()
        ```
    """

    def __init__(
        self,
        query: str,
        index_name: str,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        use_critic_agent: Optional[bool] = True,
        stream: bool = False,
        llm_provider: Optional[object] = None,
        use_graph_rag: Optional[bool] = False,
        cache: Optional[bool] = False
    ):
        # Store parameters
        self.query = query
        self.index_name = index_name
        self.video_id = video_id
        self.url = url
        self.use_critic_agent = use_critic_agent
        self.stream = stream
        self.use_graph_rag = use_graph_rag
        self.cache = cache
        # Initialize configuration and logging
        self.config = MMCTConfig()

        # Initialize LLM provider
        self.llm_provider = llm_provider or self._create_llm_provider()

    def _create_llm_provider(self) -> object:
        """Create LLM provider from configuration."""
        return provider_factory.create_llm_provider()

    async def __call__(self) -> VideoAgentResponse:
        """
        Main execution method for the VideoAgent using Swarm orchestration.

        Returns:
            VideoAgentResponse: Structured response containing the answer to the query.
        """
        try:
            # Call the video_qna function (v2 with Swarm) with simplified parameters
            # Get response from video_qna with Swarm orchestration
            video_qna_response = await video_qna(
                query=self.query,
                video_id=self.video_id,
                url=self.url,
                use_critic_agent=self.use_critic_agent,
                index_name=self.index_name,
                stream=self.stream,
                llm_provider=self.llm_provider,
                use_graph_rag=self.use_graph_rag,
                cache = self.cache
            )

            # Generate final formatted answer using LLM with video_qna response
            formatted_response = await self._generate_final_answer(video_qna_response)
            return formatted_response

        except Exception as e:
            return self._create_error_response(f"VideoAgent execution failed: {str(e)}")
        finally:
            # Clean up resources
            await self.cleanup()

    async def _generate_final_answer(self, video_qna_response: dict) -> VideoAgentResponse:
        """
        Use LLM to generate a final consolidated and structured answer.

        Args:
            video_qna_response: Response from video_qna function

        Returns:
            VideoAgentResponse: Formatted response using pydantic model
        """
        try:
            # Prepare context and messages
            context_text = str(video_qna_response)
            messages = self._prepare_messages(context_text)

            # Get structured response from LLM
            response = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=self.config.llm.temperature,
                response_format=VideoAgentResponse
            )
            return response

        except Exception as e:
            return self._create_error_response(f"Error generating final answer: {str(e)}")

    def _prepare_messages(self, context_text: str) -> list:
        """Prepare messages for LLM completion."""
        return [
            {"role": "system", "content": VIDEO_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {self.query}\nContext: {context_text}"
            }
        ]

    def _create_error_response(self, error_message: str) -> VideoAgentResponse:
        """Create a structured error response."""
        return VideoAgentResponse(
            response=error_message,
            answer_found=False,
            source=[],
            tokens={"input_token": 0, "output_token": 0}
        )

    async def cleanup(self):
        """Clean up resources and close connections."""
        try:
            if self.llm_provider and hasattr(self.llm_provider, 'close'):
                await self.llm_provider.close()
        except Exception as e:
            logger.error(f"Error during VideoAgent cleanup: {e}")


if __name__ == "__main__":

    async def main():
        """Example usage of VideoAgent with Swarm orchestration."""
        query = "<placeholder for query>"
        url = "<placeholder for url>" #Optional
        index_name = "<placeholer for index name>"
        stream = False
        cache = False
        video_agent = VideoAgent(
            query=query,
            url=url,
            index_name=index_name,
            use_critic_agent=True,
            stream=stream,
            cache = cache
        )

        results = await video_agent()
        if stream:
            messages = await Console(results)
            # if isinstance(messages, TaskResult):
            #     return messages.messages[-1]
            # return messages
        else:
            print("-" * 60)
            print(f"Query: {query}")
            print("-" * 60)
            print(results)

    asyncio.run(main())