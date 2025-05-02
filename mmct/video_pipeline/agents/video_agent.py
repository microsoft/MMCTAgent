# Standard Library
import asyncio
import os
import sys
import ast
from typing import List, Optional

# Third-party
from dotenv import load_dotenv

# Local Imports
from mmct.video_pipeline.core.tools.video_qna import video_qna, VideoQnaTools
from mmct.video_pipeline.core.tools.video_search import video_search
from mmct.video_pipeline.prompts_and_description import (
    VIDEO_AGENT_SYSTEM_PROMPT,
    VideoAgentResponse,
)
from mmct.llm_client import LLMClient

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
    """

    def __init__(
        self,
        query: str,
        index_name: str,
        top_n: int = 1,
        use_azure_cv_tool: bool = True,
        use_critic_agent: Optional[bool] = True,
        stream: Optional[bool] = False,
    ) -> None:
        self.query = query
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

        service_provider = os.getenv("LLM_PROVIDER", "azure")
        self.openai_client = LLMClient(
            service_provider=service_provider, isAsync=True
        ).get_client()
        self.model_name = os.getenv(
            "AZURE_OPENAI_VISION_MODEL"
            if service_provider == "azure"
            else "OPENAI_VISION_MODEL"
        )

        self.session_results = []

    async def __call__(self) -> VideoAgentResponse:
        self.video_ids = await video_search(
            query=self.query, index_name=self.index_name, top_n=self.top_n
        )

        print(f"video ids from AI Search:{self.video_ids['video_id']}")

        self.session_results = await asyncio.gather(
            *(
                self._fetch_mmct_response(video_id)
                for video_id in self.video_ids["video_id"]
            )
        )

        return await self._generate_final_answer()

    async def _fetch_mmct_response(self, video_id: str) -> dict:
        """Fetch MMCT (video QnA) response for a single video ID."""
        response = await video_qna(
            video_id=video_id,
            query=self.query,
            tools=self.tools,
            use_critic_agent=self.use_critic_agent,
            stream=self.stream,
            use_azure_cv_tool=self.use_azure_cv_tool,
        )
        return {"video_id": video_id, "response": response}

    async def _generate_final_answer(self) -> dict:
        """Use LLM to generate a final consolidated answer."""
        print("Generating final answer...")

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


        response = await self.openai_client.beta.chat.completions.parse(
            model=self.model_name,
            temperature=0,
            messages=messages,
            response_format=VideoAgentResponse,
        )
        response_content: VideoAgentResponse = response.choices[0].message.parsed

        return response_content


if __name__ == "__main__":

    async def main():
        query = "Describe the video content and summarize the key points."
        index_name = "general-video-index-v2"
        use_azure_cv_tool = False
        stream = False
        use_critic_agent = False
        top_n = 1

        video_agent = VideoAgent(
            query = query,
            index_name = index_name,
            top_n = top_n,
            use_critic_agent = use_critic_agent,
            use_azure_cv_tool = use_azure_cv_tool,
            stream = stream,
        )
        results = await video_agent()

        print("-" * 60)
        print(results)

    asyncio.run(main())
