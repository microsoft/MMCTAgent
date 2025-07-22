"""
This tool can do the visual criticism on the provided logs containing reasoning and query.
"""

# Importing Libaries
import os
import json
import numpy as np
import asyncio
from typing_extensions import Annotated
from mmct.video_pipeline.prompts_and_description import get_critic_tool_system_prompt
from mmct.llm_client import LLMClient
from mmct.video_pipeline.utils.helper import (
    download_blobs,
    encode_image_to_base64,
    stack_images_horizontally,
    load_images,
    get_media_folder
)

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

service_provider = os.getenv("LLM_PROVIDER", "azure")
openai_client = LLMClient(service_provider=service_provider, isAsync=True)
openai_client = openai_client.get_client()


async def critic_tool(
    timestamps_predicted: Annotated[
        str,
        "A pipe-separated list of relevant timestamps (fewer than 10), in %H:%M:%S format only. Example: '00:00:00|00:01:30'. Entries like '00:00:27,920' or 'END' are invalid.\n"
        "This timestamps are at which we found the answer or information.",
    ],
    logs: Annotated[
        str,
        "A structured string representing the agentic reasoning workflow. This includes:\n"
        "- The original user query that initiated the workflow.\n"
        "- A chronological chain of reasoning steps taken by the agents.\n"
        "- Tool invocation records (tool name, inputs, and outputs).\n"
        "- Intermediate observations and reflections.\n"
        "This log serves as a trace of how the final output was produced, useful for critique and debugging.",
    ],
    video_id: Annotated[
        str, "A unique identifier representing the video being analyzed."
    ],
    use_computer_vision_tool: Annotated[
        bool, "Flag indicating whether the Computer Vision service was used."
    ],
):
    """
    A critique tool used by the Critic Agent to evaluate the correctness, coherence, and quality of
    the planner response, based on reasoning logs and predicted timestamps.

    Parameters:
    - timestamps_predicted (str): A pipe-separated list of timestamps in strict %H:%M:%S format.
      The list must contain fewer than 10 timestamps. Example: '00:00:00|00:01:30'.
      Invalid formats such as milliseconds or keywords like 'END' are not allowed.

    - logs (str): A comprehensive, structured log of the agent's reasoning and workflow chain.
      This should include:
        * The initial user query.
        * A sequence of reasoning steps performed by the agent.
        * Each tool used, along with input parameters and their respective outputs.
        * Any intermediate insights or reflections.
      This log enables detailed critique of the decision-making and output-generation process.

    - video_id (str): The identifier of the video under evaluation.

    - use_computer_vision_tool (bool): Indicates whether Computer Vision API was leveraged
      to assist in the analysis or critique process.
    """
    try:
        timestamps = timestamps_predicted.split("|")
        if not timestamps or timestamps[0] == "END":
            return "Invalid timestamps,please provide `timestamps_predicted` in correct format."
        total_frames, base64Frames = 10, []
        frames_per_timestamp = total_frames // len(timestamps)
        assignment_str, assignment_idx = "", 0
        frame_indices = []

        for _, timestamp in enumerate(timestamps):
            h, m, s = map(int, timestamp.split(":"))
            index_of_frame = int(h * 3600 + m * 60 + s)
            delta = frames_per_timestamp / 2
            lower_bound = [
                index_of_frame - (i + 1) for i in range(int(np.floor(delta)))
            ] or [index_of_frame - 1]
            upper_bound = [
                index_of_frame + (i + 1) for i in range(int(np.ceil(delta)))
            ] or [index_of_frame + 1]
            frame_indices = list(set(frame_indices + lower_bound + upper_bound))

        frame_indices = sorted(set(frame_indices))
        frame_indices = [idx for idx in frame_indices if idx >= 0]

        # Prepare download paths
        base_dir = os.path.join(await get_media_folder(),"Frames",f"{video_id}")
        frames_download_path = base_dir
        os.makedirs(frames_download_path, exist_ok=True)

        # Download blobs
        blob_names = [f"frames/{video_id}/frame_{i}.png" for i in frame_indices]
        downloaded_files = await download_blobs(
            blob_names=blob_names, output_dir=frames_download_path
        )
        images = await load_images(
            file_paths=[
                os.path.join(frames_download_path, file) for file in downloaded_files
            ]
        )

        # Stack images
        selected_frames = images if len(images) < total_frames else []
        if len(images) > total_frames:
            stacked_images = []
            stack_size = len(frame_indices) // len(timestamps)
            for i in range(0, len(images), stack_size):
                image_chunk = images[i : i + stack_size]
                if len(image_chunk) == stack_size:
                    stacked_image = await stack_images_horizontally(
                        image_chunk, type="image"
                    )
                    stacked_images.append(stacked_image)
            selected_frames = stacked_images

        base64Frames = [
            await encode_image_to_base64(image=img) for img in selected_frames
        ]
        assigned_frame_numbers = [
            str(i + 1)
            for i in range(assignment_idx, assignment_idx + len(selected_frames))
        ]
        assignment_idx += len(selected_frames)
        assignment_str += f"Image(s) {', '.join(assigned_frame_numbers)} are for timestamp {timestamp}; "

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{i}"}}
            for i in base64Frames
        ]
        content.append({"type": "text", "text": f"These are the logs: {logs}"})
        content.append({"type": "text", "text": assignment_str})

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": await get_critic_tool_system_prompt(
                                use_computer_vision_tool=use_computer_vision_tool
                            ),
                        }
                    ],
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "top_p": 0.1,
        }

        retry_intervals = [10, 15]
        for attempt, wait_time in enumerate(retry_intervals, start=1):
            try:
                response = await openai_client.chat.completions.create(
                    model=os.getenv(
                        "LLM_VISION_DEPLOYMENT_NAME"
                        if os.getenv("LLM_PROVIDER") == "azure"
                        else "OPENAI_VISION_MODEL_NAME"
                    ),
                    temperature=payload["temperature"],
                    messages=payload["messages"],
                    top_p=payload["top_p"],
                )
                break
            except Exception as e:
                if attempt < len(retry_intervals):
                    print(
                        f"Attempt {attempt} failed: {e}. Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    return f"Final attempt failed: {e}"

        response_content = json.loads(response.choices[0].message.content)
        return json.dumps(
            {"Critic Feedback": response_content.get("Feedback", "")}
        ), response_content.get("Verdict", "YES")
    except Exception as e:
        raise Exception(e)


if __name__ == "__main__":
    # Example usage - replace with your actual values
    timestamps_predicted = "00:00:10|00:00:30|00:01:00|00:01:30|00:02:00|00:02:30"
    video_id = "example_video_id_hash"
    logs = "query: example question about the video, response: example response about the video content"
    use_computer_vision_tool = True
    res = asyncio.run(
        critic_tool(
            timestamps_predicted=timestamps_predicted,
            video_id=video_id,
            logs=logs,
            use_computer_vision_tool=use_computer_vision_tool,
        )
    )
    print(res)
