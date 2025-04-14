"""
Main script where agents are initialised and configured
"""

# Importing modules
import asyncio
import os
import sys
from typing_extensions import Annotated

import torch

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.base import TaskResult

from mmct.image_pipeline.core.tools.vit import VITTool
from mmct.image_pipeline.core.tools.recog import RECOGTool
from mmct.image_pipeline.core.tools.object_detect import ObjectDetectTool
from mmct.image_pipeline.core. tools.ocr import OCRTool
from mmct.image_pipeline.core.tools.critic import criticTool
from mmct.image_pipeline.core.tools.tools_description import *

from mmct.image_pipeline.prompts import (
    get_planner_system_prompt,
    get_critic_system_prompt,
    get_critic_tool_prompt,
)

from mmct.llm_client import LLMClient

from dotenv import load_dotenv

load_dotenv(override=True)

# Getting LLM client for autogen
service_provider = os.getenv("LLM_PROVIDER", "azure")
model_client = LLMClient(autogen=True, service_provider=service_provider)
model_client = model_client.get_client()


async def run_task(
    image_path: str,
    query: str,
    tools: str,
    criticFlag: str = "True",
    stream: Annotated[bool, "Enable streaming response (True/False)"] = True,
):
    criticFlag = True if criticFlag.lower()=="true" else False
    torch.cuda.empty_cache()  # clearing GPU memory
    tools = tools.split(",")
    tools_string = ", ".join(tools)

    available_tools = {
        "QueryObjectDetectTool": ObjectDetectTool,
        "QueryOCRTool": OCRTool,
        "QueryVITTool": VITTool,
        "QueryRECOGTool": RECOGTool,
    }

    tools_list = [available_tools[tool] for tool in tools if tool in available_tools]

    # Define Planner agent
    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        model_client_stream=False,
        system_message=await get_planner_system_prompt(
            tools_string=tools_string, criticFlag=criticFlag
        ),
        tools=tools_list,
        reflect_on_tool_use=True,
    )

    if criticFlag:
        critic = AssistantAgent(
            name="critic",
            model_client=model_client,
            model_client_stream=False,
            system_message=await get_critic_system_prompt(includeMetaGuidelines=True),
            tools=[criticTool],
            reflect_on_tool_use=False,
        )

        text_mention_termination = TextMentionTermination("TERMINATE")
        max_messages_termination = MaxMessageTermination(max_messages=20)
        termination = text_mention_termination | max_messages_termination

        team = SelectorGroupChat(
            [planner, critic],
            model_client=model_client,
            termination_condition=termination,
            allow_repeated_speaker=True,
        )

        task = f"query:{query},image path:{image_path}. {'always criticize the final response if planner asks for review and provide feedback on the final response.' if criticFlag else ''}"
        if stream:
            return team.run_stream(task=task)
        else:
            result = await team.run(task=task)
            return result.messages[-1]
        
    else:
        task = f"query:{query},image path:{image_path}."
        if stream:
            return planner.run_stream(task=task)
        else:
            result = await planner.run(task=task)
            return result.messages[-1]


if __name__ == "__main__":
    image_path = "C:/Users/v-amanpatkar/Downloads/menu.png"
    query = """I want to order House Special & Crab Curry. What will be the order amount?
        In this same order amount what are the option that I can consider."""
    tools = "QueryVITTool,QueryObjectDetectTool"
    criticFlag = 'False'
    stream = True
    if stream:

        async def run():
            stream_response = await run_task(
                image_path=image_path, query=query, criticFlag=criticFlag, tools=tools
            )

            async for response in stream_response:
                if isinstance(response,TaskResult):
                        continue
                print("\n", response, flush=True)  # Streaming output in real-time

        asyncio.run(run())

    else:

        async def fetch():
            return await run_task(
                image_path=image_path,
                query=query,
                criticFlag=criticFlag,
                tools=tools,
                stream=stream,
            )

        response = asyncio.run(fetch())
        print(response)
