"""
This tool can do the sementical query on the summary_n_transcript file and provide the top 3 timestamps.
"""

# Importing Libraries
import json
import os
from scipy.spatial.distance import cosine
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import Annotated
from datetime import timedelta
from datetime import datetime
from mmct.llm_client import LLMClient
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from mmct.video_pipeline.utils.helper import get_media_folder
# Load environment variables
load_dotenv(find_dotenv(), override=True)

service_provider = os.getenv("LLM_PROVIDER", "azure")
client_embed = LLMClient(
    service_provider=service_provider, embedding=True, isAsync=True
)
client_embed = client_embed.get_client()


async def query_summary_n_transcript(
    summary_query: Annotated[str, "query summary & transcript"],
    video_id: Annotated[str, "video id"],
) -> str:
    try:
        logger.info("Initialization of query summary and transcript tool")
        base_dir = await get_media_folder()
        os.makedirs(base_dir, exist_ok=True)
        summary_path = os.path.join(
            base_dir,
            f"{video_id}.json",
        )

        async def average_time(time_str1, time_str2):
            # Convert time strings to timedelta objects
            t1 = timedelta(
                hours=int(time_str1.split(":")[0]),
                minutes=int(time_str1.split(":")[1]),
                seconds=int(time_str1.split(":")[2]),
            )
            t2 = timedelta(
                hours=int(time_str2.split(":")[0]),
                minutes=int(time_str2.split(":")[1]),
                seconds=int(time_str2.split(":")[2]),
            )

            # Calculate the average
            avg_time = (t1 + t2) / 2

            # Convert the average timedelta back to time format
            total_seconds = int(avg_time.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)

        async def extract_timestamped_data(text_block):
            """Extract timestamps and text from a text block formatted as '[timestamp] text'."""
            extracted_data = []
            descriptions = text_block.strip().split(
                "\n\n"
            )  # Split into individual descriptions

            for desc in descriptions:
                lines = desc.split("\n")
                if len(lines) > 1:
                    timestamp = lines[0].strip("[]")  # Extract timestamp
                    text = " ".join(lines[1:]).strip()
                    extracted_data.append({"timestamp": timestamp, "text": text})

            return extracted_data

        # Load JSON file
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data is None:
            logger.error(f"File not found: {summary_path}")

        # Extract data from "summaries" and "transcript"
        summaries_data, transcript_data = [], []
        logger.info(
            "Checking for the valid content in the summary and transcript content."
        )
        if "summaries" in data:
            if "detailed_description" in data["summaries"]:
                summaries_data.extend(
                    await extract_timestamped_data(
                        data["summaries"]["detailed_description"]
                    )
                )
            if "Action_taken" in data["summaries"]:
                summaries_data.extend(
                    await extract_timestamped_data(data["summaries"]["Action_taken"])
                )

        if "transcript" in data:
            lines = data["transcript"].strip().split("\n")
            for i in range(1, len(lines), 2):  # Every 2 lines make one record
                if i + 1 < len(lines):
                    timestamp_text = lines[i + 1].strip()
                    timestamp_range = lines[i].split("-->")  # Extract start & end time
                    if len(timestamp_range) == 2:
                        start_timestamp = timestamp_range[0].strip()  # Take start time
                        end_timestamp = timestamp_range[1].strip()

                        # Remove any extra data (milliseconds, microseconds) before converting to datetime
                        start_timestamp = start_timestamp.split(",")[
                            0
                        ]  # Remove milliseconds
                        end_timestamp = end_timestamp.split(",")[
                            0
                        ]  # Remove milliseconds

                        # Format timestamps to HH:MM:SS before passing to average_time
                        start_timestamp = datetime.strptime(
                            start_timestamp, "%H:%M:%S"
                        ).strftime("%H:%M:%S")
                        end_timestamp = datetime.strptime(
                            end_timestamp, "%H:%M:%S"
                        ).strftime("%H:%M:%S")

                        timestamp = await average_time(start_timestamp, end_timestamp)

                        transcript_data.append(
                            {"timestamp": timestamp, "text": timestamp_text}
                        )

        # Combine all extracted text and timestamps
        all_data = summaries_data + transcript_data
        all_data = [
            i for i in all_data if ("text" in i and i["text"] and len(i["text"]) != 0)
        ]  # removing null and empty text containing dictionaries

        async def get_embeddings_batch(texts, model, batch_size=32):
            """Get embeddings for a batch of texts."""
            processed_texts = [text.replace("\n", " ") for text in texts]
            embeddings = []
            processed_texts = [i for i in processed_texts if (i and len(i) != 0)]
            print("Processed texts", processed_texts)
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i : i + batch_size]
                response = await client_embed.embeddings.create(
                    input=batch, model=model
                )
                embeddings.extend([data.embedding for data in response.data])

            return embeddings

        # Generate embeddings for all extracted texts in batch
        texts = [entry["text"] for entry in all_data]
        logger.info("Embedding creation initiated")
        embeddings = await get_embeddings_batch(
            texts,
            model=os.getenv(
                "EMBEDDING_SERVICE_MODEL_NAME"
                if os.getenv("LLM_PROVIDER") == "azure"
                else "OPENAI_EMBEDDING_MODEL_NAME"
            ),
        )
        # Assign embeddings to data
        for i, entry in enumerate(all_data):
            entry["embedding"] = embeddings[i]

        # Get embedding for query
        query_embedding = (
            (
                await client_embed.embeddings.create(
                    input=[summary_query],
                    model=os.getenv(
                        "EMBEDDING_SERVICE_MODEL_NAME"
                        if os.getenv("LLM_PROVIDER") == "azure"
                        else "OPENAI_EMBEDDING_MODEL_NAME"
                    ),
                )
            )
            .data[0]
            .embedding
        )

        # Cosine Similarity
        def calculate_similarity(embedding1, embedding2):
            """calculation of cosine similarity."""
            return cosine(np.array(embedding1), np.array(embedding2))

        # Use ThreadPoolExecutor to compute cosine similarity in parallel
        with ThreadPoolExecutor() as executor:
            similarities = list(
                executor.map(
                    lambda entry: calculate_similarity(
                        query_embedding, entry["embedding"]
                    ),
                    all_data,
                )
            )
        logger.info("Similarity calculated")
        # Get top 3 closest matches
        top_3_indices = np.argsort(similarities)[:3]
        top_3_timestamps = [all_data[i]["timestamp"] for i in top_3_indices]

        return ", ".join(top_3_timestamps)

    except Exception as e:
        raise Exception(f"Error in query_summary_n_transcript tool: {e}")
