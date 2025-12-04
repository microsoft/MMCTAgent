# Standard Library
import asyncio
from typing import Optional
from dotenv import load_dotenv

# Local Imports
from mmct.video_pipeline.core.tools.video_qna import video_qna
from mmct.video_pipeline.prompts_and_description import (
    VIDEO_AGENT_SYSTEM_PROMPT,
    VideoAgentResponse,
)
from autogen_agentchat.ui import Console
from mmct.config.providers import VideoAgentProviderConfig

# Load environment variables
load_dotenv(override=True)


class VideoAgent:
    """
    MMCT's Video question answering agent using Swarm orchestration.
    
    This agent uses VideoAgentProviderConfig for dependency injection to access:
    - llm_provider: For LLM-based reasoning and structured response generation
    - vectordb_chapter: For retrieving video context and transcripts
    - vectordb_object_registry: For retrieving video summaries and object collections
    - vectordb_keyframes: For searching relevant video frames
    - embedding_provider: For generating embeddings for semantic search
    - storage_provider: For accessing stored video frames

    This agent provides a clean interface that:
    1. Calls video_qna (with Swarm orchestration) with the provided parameters
    2. Formats the response using LLM with structured output
    3. Returns a properly structured VideoAgentResponse

    Args:
        query (str): The natural language question about video content.
        provider (VideoAgentProviderConfig): Provider configuration containing all required providers.
        video_id (Optional[str]): Specific video ID to query. Defaults to None.
        url (Optional[str]): URL to filter the search results for that particular video. Defaults to None.
        use_critic_agent (bool): Whether to use the critic agent for validation. Defaults to True.
        stream (bool): Whether to stream the response output. Defaults to False.
        cache (bool): Whether to enable caching for model responses. Defaults to False.

    Example:
        Basic usage with query and index:
        ```python
        from mmct.config.providers import VideoAgentProviderConfig
        from mmct.providers.azure import (
            AzureLLMProvider,
            AzureEmbeddingProvider,
            AISearchChapterProvider,
            AISearchKeyframesProvider,
            AISearchObjectCollectionProvider
            AzureStorageProvider,
        )
        # Note: Image Embedding provider is also required which is clip based provider.
        from mmct.providers.local import ClipImageEmbeddingProvider


        # Initialize all required providers
        provider = VideoAgentProviderConfig(
            llm_provider=AzureOpenAILLMProvider(endpoint = "<some-endpoint>",api_version="<api-version>",...),
            embedding_provider=AzureOpenAIEmbeddingProvider(...),
            vectordb_chapter=AISearchChapterProvider(...),
            vectordb_object_registry=AISearchObjectCollectionProvider(...),
            vectordb_keyframes=AISearchKeyframesProvider(...),
            storage_provider=AzureBlobStorageProvider(...),
            image_embedding_provider=ClipImageEmbeddingProvider(...)
        )
        
        video_agent = VideoAgent(
            query="What are the benefits of organic farming?",
            provider=provider
        )
        result = await video_agent()
        print(result.response)
        ```

        With specific video ID:
        ```python
        video_agent = VideoAgent(
            query="Explain the farming technique shown",
            provider=provider,
            video_id="abc123def456"
        )
        result = await video_agent()
        ```

        With URL and streaming:
        ```python
        video_agent = VideoAgent(
            query="Summarize this farming video",
            provider=provider,
            url="https://video-url.mp4",
            stream=True
        )
        result = await video_agent()
        ```
    """

    def __init__(
        self,
        query: str,
        provider: VideoAgentProviderConfig,
        video_id: Optional[str] = None,
        url: Optional[str] = None,
        use_critic_agent: Optional[bool] = True,
        stream: bool = False,
        cache: Optional[bool] = False
    ):
        # Store parameters
        self.query = query
        self.video_id = video_id
        self.url = url
        self.use_critic_agent = use_critic_agent
        self.stream = stream
        self.cache = cache
        self.provider = provider


    async def __call__(self) -> VideoAgentResponse:
        """
        Main execution method for the VideoAgent using Swarm orchestration.

        Returns:
            VideoAgentResponse: Structured response containing the answer to the query.
        """
        try:
            # Call the video_qna function
            # Get response from video_qna with Swarm orchestration
            video_qna_response = await video_qna(
                query=self.query,
                video_id=self.video_id,
                url=self.url,
                use_critic_agent=self.use_critic_agent,
                stream=self.stream,
                provider = self.provider,
                cache = self.cache
            )

            # Generate final formatted answer using LLM with video_qna response
            formatted_response = await self._generate_final_answer(video_qna_response)
            return formatted_response

        except Exception as e:
            return self._create_error_response(f"VideoAgent execution failed: {str(e)}")
        finally:
            pass

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
            response = await self.provider.llm_provider.chat_completion(
                messages=messages,
                temperature=0.0,  # Use default temperature
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


if __name__ == "__main__":

    async def main():
        """Example usage of VideoAgent with Swarm orchestration."""
        query = "<placeholder for query>"
        url = "<placeholder for url>" #Optional
        stream = False
        cache = False
        video_agent = VideoAgent(
            query=query,
            url=url,
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