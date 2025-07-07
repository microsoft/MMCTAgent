# Standard Library
import asyncio
import os
from typing import Optional, Annotated
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
# Local Imports
from mmct.video_pipeline.core.tools.video_qna import video_qna, VideoQnaTools
from mmct.video_pipeline.core.tools.video_search import video_search
from mmct.video_pipeline.prompts_and_description import (
    VIDEO_AGENT_SYSTEM_PROMPT,
    VideoAgentResponse,
)
from mmct.video_pipeline.utils.helper import remove_file
from mmct.llm_client import LLMClient
from mmct.custom_logger import log_manager
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
    Optional modules like Azure Computer Vision and a critic agent can be included in the pipeline.

    Args:
        query (str): The natural language question related to the video content.
        index_name (str): Name of the Azure Cognitive Search index used to retrieve relevant videos.
        top_n (int, optional): Number of top relevant videos to fetch. A higher value increases
            processing time since MMCT runs on each fetched video. Defaults to 1.
        use_azure_cv_tool (bool, optional): Whether to use Azure Computer Vision for visual content analysis. Defaults to True.
        use_critic_agent (bool, optional): Whether to use the critic agent as part of the MMCT pipeline for answer validation or refinement. Defaults to True.
        stream (bool, optional): Whether to stream the response output. Defaults to False.
        disable_console_log (bool):
            Boolean flag to disable console logs. Default set to False.
    """

    def __init__(
        self,
        query: str,
        index_name: str,
        video_id: Optional[str]=None,
        top_n: int = 1,
        use_azure_cv_tool: bool = True,
        use_critic_agent: Optional[bool] = True,
        stream: Optional[bool] = False,
        disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False
    ):
        try:
            self.query = query
            self.video_id = video_id
            self.index_name = index_name
            self.top_n = top_n
            self.use_azure_cv_tool = use_azure_cv_tool
            self.tools = [
                VideoQnaTools.GET_SUMMARY_WITH_TRANSCRIPT,
                VideoQnaTools.QUERY_SUMMARY_TRANSCRIPT,
                VideoQnaTools.QUERY_GPT_VISION,
            ]
            if self.use_azure_cv_tool:
                self.tools.append(VideoQnaTools.QUERY_FRAMES_AZURE_COMPUTER_VISION)
                
            self.tools = [str(tool.name) for tool in self.tools]
            self.use_critic_agent = use_critic_agent
            self.stream = stream
            if disable_console_log==False:
                    log_manager.enable_console()
            else:
                log_manager.disable_console()
            self.logger = log_manager.get_logger()

            service_provider = os.getenv("LLM_PROVIDER", "azure")
            self.openai_client = LLMClient(
                service_provider=service_provider, isAsync=True
            ).get_client()
            self.model_name = os.getenv(
                "LLM_VISION_MODEL_NAME"
                if service_provider == "azure"
                else "OPENAI_VISION_MODEL_NAME"
            )
            self.logger.info("Initialized the llm model client")
            self.session_results = []
        except Exception as e:
            self.logger.exception(f"Exception occured while initializing the Video Agent: {e}")
            raise
        
    async def fetch_metadata_ai_index(self, index_name, hash_video_id):
        try:
            AZURE_MANAGED_IDENTITY = os.environ.get("MANAGED_IDENTITY", None)
            azure_search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT", None)

            if azure_search_endpoint is None:
                raise Exception("Azure search endpoint is missing in env!")

            if AZURE_MANAGED_IDENTITY is None:
                raise Exception(
                    "MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )

            # Setup credentials
            if AZURE_MANAGED_IDENTITY.upper() == "TRUE":
                credential = DefaultAzureCredential()
            else:
                SEARCH_SERVICE_KEY = os.environ.get("SEARCH_SERVICE_KEY", None)
                if SEARCH_SERVICE_KEY is None:
                    raise Exception("Azure Search Key is missing!")
                credential = AzureKeyCredential(SEARCH_SERVICE_KEY)

            # Create async search client
            search_client = SearchClient(
                endpoint=azure_search_endpoint,
                index_name=index_name,
                credential=credential,
            )

            # Perform async search with filter
            results = await search_client.search(
                search_text="*",
                filter=f"hash_video_id eq '{hash_video_id}'",
                select=["youtube_url","blob_video_url","hash_video_id"]
            )

            youtube_urls = []
            async for result in results:
                print(result)
                metadata = {"video_id": [result.get("hash_video_id")], "video_url": [{"BLOB":result.get("blob_video_url"),"YT_URL":result.get("youtube_url")}]}
                #youtube_urls.append(result.get("youtube_url"))
            
            if search_client:
                await search_client.close()
            if isinstance(credential, DefaultAzureCredential):
                credential.close()
                

            return metadata#youtube_urls[0] if (isinstance(youtube_urls,list) and len(youtube_urls)!=0) else None

        except Exception as e:
            raise


    async def __call__(self) -> VideoAgentResponse:
        try:
            if self.video_id:
                self.video_ids = await self.fetch_metadata_ai_index(hash_video_id=self.video_id, index_name=self.index_name)
            else:
                self.logger.info("Performing the Video Search for the provided input")
                self.video_ids = await video_search(
                    query=self.query, index_name=self.index_name, top_n=self.top_n
                )
                self.logger.info("Successfully retrieved the results from the Video Search")
                
                filtered_pairs = [
                    (vid, url) for vid, url in zip(self.video_ids["video_id"], self.video_ids["video_url"])
                ]
                

                # Rebuild the self.video_ids structure
                self.video_ids = {
                    "video_id": [vid for vid, _ in filtered_pairs],
                    "video_url": [url for _, url in filtered_pairs]
                }
                # self.logger.info("Filtered out the video ids with None or empty youtube urls")
                if len(self.video_ids['video_id'])==0:
                    return {"is_video_unavailable":True}
                self.logger.info(f"Video ids from AI Search:{self.video_ids['video_id']}")

            self.logger.info(f"Accumulating the MMCT Video Agent Response for retrieved Video Ids")
            self.session_results = await asyncio.gather(
                *(
                    self._fetch_mmct_response(video_id)
                    for video_id in self.video_ids["video_id"]
                )
            )
            self.logger.info("Successfully retrieved the results for the Video Ids")
            # Remove files
            self.logger.info("Cleaning up media files for the retrieved Video Ids")
            await asyncio.gather(
                *(
                    remove_file(video_id)
                    for video_id in self.video_ids["video_id"]
                )
            )
            self.logger.info("Media cleanup completed.")
            self.logger.info("Generating the final answer from the accumulated answers")
            return await self._generate_final_answer()
        except Exception as e:
            self.logger.exception(f"Video Agent Workflow failed: {e}")
            raise

    async def _fetch_mmct_response(self, video_id: str) -> dict:
        """Fetch MMCT (video QnA) response for a single video ID."""
        try:
            self.logger.info(f"Performing Video Q&A for \nvideo id: {video_id}\nquery:{self.query}")
            response = await video_qna(
                video_id=video_id,
                query=self.query,
                tools=self.tools,
                use_critic_agent=self.use_critic_agent,
                stream=self.stream,
                use_azure_cv_tool=self.use_azure_cv_tool,
            )
            self.logger.info(f"Video Agent's response:\nVideo Id: {video_id}\nVideo Agent Response: {response}")
            return {"video_id": video_id, "response": response}
        except Exception as e:
            self.logger.exception(f"Exception occured while fetching MMCT Video Agent Response for:\nVideo Id: {video_id}\nQuery: {self.query}\nException: {e}")
            raise

    async def _generate_final_answer(self) -> dict:
        """Use LLM to generate a final consolidated answer."""
        try:
            self.logger.info("Generating the final answer for the acccumulated results...")
            self.logger.info("Preparing the messages payload")
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
            self.logger.info("Successfully prepared the messages payload")
            self.logger.info("Generating the LLM Response")
            response = await self.openai_client.beta.chat.completions.parse(
                model=self.model_name,
                temperature=0,
                messages=messages,
                response_format=VideoAgentResponse,
            )
            self.logger.info("Successfully generated the LLM Response")
            response_content: VideoAgentResponse = response.choices[0].message.parsed
            self.logger.info("Converting the response to structured pydantic response")
            return response_content
        except Exception as e:
            raise

if __name__ == "__main__":

    async def main():
        query = "What is the neemastram?"
        index_name = "telugu-video-index"
        use_azure_cv_tool = False
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
            use_azure_cv_tool = use_azure_cv_tool,
            stream = stream,
        )
        results = await video_agent()

        print("-" * 60)
        print(results)

    asyncio.run(main())
