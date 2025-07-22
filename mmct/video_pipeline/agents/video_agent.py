# Standard Library
import asyncio
from typing import Optional, Annotated
from dotenv import load_dotenv

# Local Imports
from mmct.video_pipeline.core.tools.video_qna import video_qna, VideoQnaTools
from mmct.video_pipeline.core.tools.video_search import video_search
from mmct.video_pipeline.prompts_and_description import (
    VIDEO_AGENT_SYSTEM_PROMPT,
    VideoAgentResponse,
)
from mmct.video_pipeline.utils.helper import remove_file
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from mmct.exceptions import ProviderException, ConfigurationException
from mmct.utils.logging_config import LoggingConfig
from mmct.utils.error_handler import handle_exceptions
from loguru import logger

# Load environment variables
load_dotenv(override=True)

# Set UTF-8 encoding globally
# sys.stdin.reconfigure(encoding="utf-8")
# sys.stdout.reconfigure(encoding="utf-8")


class VideoAgent:
    """
    Agent for performing question answering based on video content, powered by MMCT (Multi-modal Critical Thinking) agents.

    This agent first fetches videos relevant to the input query using an Azure Cognitive Search index.
    It then applies the MMCT pipeline on the selected videos to generate answers.
    Optional modules like Computer Vision and a critic agent can be included in the pipeline.

    Args:
        query (str): The natural language question related to the video content.
        index_name (str): Name of the Azure Cognitive Search index used to retrieve relevant videos.
        top_n (int, optional): Number of top relevant videos to fetch. A higher value increases
            processing time since MMCT runs on each fetched video. Defaults to 1.
        use_computer_vision_tool (bool, optional): Whether to use Computer Vision for visual content analysis. Defaults to True.
        use_critic_agent (bool, optional): Whether to use the critic agent as part of the MMCT pipeline for answer validation or refinement. Defaults to True.
        stream (bool, optional): Whether to stream the response output. Defaults to False.
        disable_console_log (bool):
            Boolean flag to disable console logs. Default set to False.
    """

    def __init__(
        self,
        query: str,
        index_name: str,
        video_id: Optional[str] = None,
        top_n: int = 1,
        use_computer_vision_tool: bool = True,
        use_critic_agent: Optional[bool] = True,
        stream: Optional[bool] = False,
        disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False
    ):
        try:
            # Initialize configuration
            self.config = MMCTConfig()
            
            # Setup logging
            LoggingConfig.setup_logging(
                level=self.config.logging.level,
                log_file=self.config.logging.log_file if self.config.logging.enable_file_logging else None,
                enable_json=self.config.logging.enable_json
            )
            
            # Initialize providers
            self.llm_provider = provider_factory.create_llm_provider(
                self.config.llm.provider,
                self.config.llm.model_dump()
            )
            self.search_provider = provider_factory.create_search_provider(
                self.config.search.provider,
                self.config.search.model_dump()
            )
            self.vision_provider = provider_factory.create_vision_provider(
                self.config.llm.provider,  # Use same provider as LLM for vision
                self.config.llm.model_dump()
            )
            
            # Set instance attributes
            self.query = query
            self.video_id = video_id
            self.index_name = index_name
            self.top_n = top_n
            self.use_computer_vision_tool = use_computer_vision_tool
            self.tools = [
                VideoQnaTools.GET_VIDEO_DESCRIPTION,
                VideoQnaTools.QUERY_VIDEO_DESCRIPTION,
                VideoQnaTools.QUERY_VISION_LLM,
            ]
            if self.use_computer_vision_tool:
                self.tools.append(VideoQnaTools.QUERY_FRAMES_COMPUTER_VISION)
                
            self.tools = [str(tool.name) for tool in self.tools]
            self.use_critic_agent = use_critic_agent
            self.stream = stream
            
            # Configure console logging
            if not disable_console_log:
                logger.enable("mmct")
            else:
                logger.disable("mmct")
            
            self.model_name = self.config.llm.model_name
            logger.info("Initialized VideoAgent with provider system")
            self.session_results = []
            
        except Exception as e:
            logger.exception(f"Exception occurred while initializing the Video Agent: {e}")
            raise ConfigurationException(f"Failed to initialize VideoAgent: {e}")
        
    @handle_exceptions(retries=3)
    async def fetch_metadata_ai_index(self, index_name: str, hash_video_id: str) -> dict:
        """
        Fetch video metadata from AI search index using the provider system.
        
        Args:
            index_name: Name of the search index
            hash_video_id: Hash of the video ID to search for
            
        Returns:
            Dictionary containing video metadata
            
        Raises:
            ProviderException: If search operation fails
        """
        try:
            # Use the search provider to perform the search
            search_query = "*"
            filter_query = f"hash_video_id eq '{hash_video_id}'"
            
            results = await self.search_provider.search(
                query=search_query,
                index_name=index_name,
                filter=filter_query,
                select=["youtube_url", "blob_video_url", "hash_video_id"]
            )
            
            if not results:
                logger.warning(f"No results found for video ID: {hash_video_id}")
                return {"video_id": [], "video_url": []}
            
            # Process results
            result = results[0]  # Take the first result
            metadata = {
                "video_id": [result.get("hash_video_id")],
                "video_url": [{
                    "BLOB": result.get("blob_video_url"),
                    "YT_URL": result.get("youtube_url")
                }]
            }
            
            logger.info(f"Successfully fetched metadata for video ID: {hash_video_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata for video ID {hash_video_id}: {e}")
            raise ProviderException(f"Search operation failed: {e}", "SEARCH_FAILED")


    @handle_exceptions(retries=2)
    async def __call__(self) -> VideoAgentResponse:
        """
        Main execution method for the VideoAgent.
        
        Returns:
            VideoAgentResponse containing the processed video analysis results
            
        Raises:
            ProviderException: If any provider operation fails
        """
        try:
            if self.video_id:
                logger.info(f"Fetching metadata for specific video ID: {self.video_id}")
                self.video_ids = await self.fetch_metadata_ai_index(
                    hash_video_id=self.video_id, 
                    index_name=self.index_name
                )
            else:
                logger.info("Performing video search for the provided input")
                self.video_ids = await video_search(
                    query=self.query, 
                    index_name=self.index_name, 
                    top_n=self.top_n,
                    search_provider=self.search_provider
                )
                logger.info("Successfully retrieved the results from the video search")
                
                # Filter out invalid video IDs
                filtered_pairs = [
                    (vid, url) for vid, url in zip(self.video_ids["video_id"], self.video_ids["video_url"])
                    if vid and url
                ]
                
                # Rebuild the self.video_ids structure
                self.video_ids = {
                    "video_id": [vid for vid, _ in filtered_pairs],
                    "video_url": [url for _, url in filtered_pairs]
                }
                
                if len(self.video_ids['video_id']) == 0:
                    logger.warning("No valid video IDs found")
                    return {"is_video_unavailable": True}
                
                logger.info(f"Video IDs from AI Search: {self.video_ids['video_id']}")

            logger.info("Accumulating the MMCT Video Agent Response for retrieved Video IDs")
            self.session_results = await asyncio.gather(
                *(
                    self._fetch_mmct_response(video_id)
                    for video_id in self.video_ids["video_id"]
                )
            )
            logger.info("Successfully retrieved the results for the Video IDs")
            
            # Remove files
            logger.info("Cleaning up media files for the retrieved Video IDs")
            await asyncio.gather(
                *(
                    remove_file(video_id)
                    for video_id in self.video_ids["video_id"]
                )
            )
            logger.info("Media cleanup completed.")
            
            logger.info("Generating the final answer from the accumulated answers")
            return await self._generate_final_answer()
            
        except Exception as e:
            logger.exception(f"Video Agent Workflow failed: {e}")
            raise ProviderException(f"VideoAgent execution failed: {e}", "AGENT_EXECUTION_FAILED")

    @handle_exceptions(retries=2)
    async def _fetch_mmct_response(self, video_id: str) -> dict:
        """
        Fetch MMCT (video QnA) response for a single video ID.
        
        Args:
            video_id: The video ID to process
            
        Returns:
            Dictionary containing video ID and response
            
        Raises:
            ProviderException: If video QnA processing fails
        """
        try:
            logger.info(f"Performing Video Q&A for video id: {video_id}")
            response = await video_qna(
                video_id=video_id,
                query=self.query,
                tools=self.tools,
                use_critic_agent=self.use_critic_agent,
                stream=self.stream,
                use_computer_vision_tool=self.use_computer_vision_tool,
                llm_provider=self.llm_provider,
                vision_provider=self.vision_provider
            )
            logger.info(f"Video Agent's response for video {video_id}: {response}")
            return {"video_id": video_id, "response": response}
        except Exception as e:
            logger.exception(f"Exception occurred while fetching MMCT Video Agent Response for video {video_id}: {e}")
            raise ProviderException(f"Video QnA processing failed for {video_id}: {e}", "VIDEO_QNA_FAILED")

    @handle_exceptions(retries=2)
    async def _generate_final_answer(self) -> dict:
        """
        Use LLM to generate a final consolidated answer.
        
        Returns:
            Dictionary containing the final consolidated response
            
        Raises:
            ProviderException: If LLM generation fails
        """
        try:
            logger.info("Generating the final answer for the accumulated results...")
            logger.info("Preparing the messages payload")
            
            messages = [
                {"role": "system", "content": VIDEO_AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: {self.query}"},
                        {"type": "text", "text": f"Context: {self.session_results}"},
                        {"type": "text", "text": f"metadata: {self.video_ids}"},
                    ],
                },
            ]
            
            logger.info("Successfully prepared the messages payload")
            logger.info("Generating the LLM Response")
            
            # Use the provider system for LLM completion
            response = await self.llm_provider.chat_completion(
                messages=messages,
                temperature=self.config.llm.temperature,
                response_format=VideoAgentResponse
            )
            
            logger.info("Successfully generated the LLM Response")
            logger.info("Converting the response to structured pydantic response")
            
            return response
            
        except Exception as e:
            logger.exception(f"Exception occurred while generating final answer: {e}")
            raise ProviderException(f"Final answer generation failed: {e}", "FINAL_ANSWER_GENERATION_FAILED")

if __name__ == "__main__":

    async def main():
        # Example usage - replace with your actual values
        query = "example question about the video"
        index_name = "your-index-name"
        use_computer_vision_tool = False
        stream = False
        use_critic_agent = True
        top_n = 1
        video_id = ""
        video_agent = VideoAgent(
            query = query,
            index_name = index_name,
            video_id=video_id,
            top_n = top_n,
            use_critic_agent = use_critic_agent,
            use_computer_vision_tool = use_computer_vision_tool,
            stream = stream,
        )
        results = await video_agent()

        print("-" * 60)
        print(results)

    asyncio.run(main())
