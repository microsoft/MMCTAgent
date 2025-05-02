import re
import numpy as np
import os
from datetime import datetime
import math
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
from mmct.llm_client import LLMClient
# Load environment variables
load_dotenv(find_dotenv(),override=True)

gpt40_client = LLMClient(service_provider=os.getenv("LLM_PROVIDER", "azure"), isAsync=True).get_client()
async def process_transcript(srt_text: str, SIMILARITY_THRESHOLD=0.7, TIME_LIMIT=50):
    """
    Parses an SRT transcript, removes redundant chunks (e.g., "Music" segments), and performs semantic chunking based on content similarity.

    Parameters:
        srt_text (str): The transcript in SRT format.
        gpt40_client (): AsyncAzureOpenAI class instance.
        SIMILARITY_THRESHOLD (float): Threshold for cosine similarity to determine chunk separation.
        TIME_LIMIT (int): Maximum duration (in seconds) of each chunk.

    Returns:
        str: A formatted string with time ranges and semantically chunked text.
    """

    async def parse_timestamp(timestamp):
        h, m, s, ms = map(int, re.split("[:,]", timestamp))
        return h * 3600 + m * 60 + s + ms / 1000

    async def format_timestamp(seconds):
        ms = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d},{ms:03d}"

    async def parse_transcript(srt_text):
        """Extracts sentences and timestamps from SRT text asynchronously."""
        entries = re.findall(
            r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\d+\n|\Z)", 
            srt_text, 
            re.DOTALL
        )
        sentences, time_stamps, end_times = [], [], []

        for index, start_time, end_time, text in entries:
            text = text.replace("\n", " ").strip()
            if text.lower() != "music" and text:
                sentences.append(text)
                time_stamps.append(await parse_timestamp(start_time))
                end_times.append(await parse_timestamp(end_time))

        return sentences, time_stamps, end_times

    async def create_embedding(text):
        """Generates an embedding for a given text using OpenAI API asynchronously."""
        try:
            response = await gpt40_client.embeddings.create(input=text, model="text-embedding-ada-002")
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {e}")

    async def calculate_cosine_similarity(vec1, vec2):
        """Computes cosine similarity between two vectors asynchronously."""
        return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]

    async def semantic_chunking(sentences, time_stamps, end_times):
        """Groups transcript text into semantic chunks based on similarity and time constraints asynchronously."""
        if not sentences:
            return ""

        chunks, current_chunk = {}, ""
        chunk_start_time = time_stamps[0]
        chunk_end_time = end_times[0]
        embeddings_cache = {}

        for i, sentence in enumerate(sentences):
            if sentence in embeddings_cache:
                current_embedding = embeddings_cache[sentence]
            else:
                current_embedding = await create_embedding(sentence)
                if current_embedding is None:
                    continue  # Skip if embedding fails
                embeddings_cache[sentence] = current_embedding

            if i > 0:
                prev_embedding = embeddings_cache[sentences[i - 1]]
                similarity = await calculate_cosine_similarity(prev_embedding, current_embedding)
                
                # Ensure chunk stays within its respective start and end time
                if similarity < SIMILARITY_THRESHOLD or end_times[i] - chunk_start_time > TIME_LIMIT:
                    if current_chunk.strip():
                        chunks[f"{await format_timestamp(chunk_start_time)} --> {await format_timestamp(chunk_end_time)}"] = current_chunk.strip()
                    current_chunk = sentence + " "  # Start new chunk
                    chunk_start_time = time_stamps[i]
                    chunk_end_time = end_times[i]
                else:
                    current_chunk += sentence + " "
                    chunk_end_time = end_times[i]

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks[f"{await format_timestamp(chunk_start_time)} --> {await format_timestamp(chunk_end_time)}"] = current_chunk.strip()

        return "\n\n".join(f"{idx+1}\n{time_range}\n{chunk}" for idx, (time_range, chunk) in enumerate(chunks.items()))

    sentences, time_stamps, end_times = await parse_transcript(srt_text)
    return await semantic_chunking(sentences, time_stamps, end_times)

async def add_empty_intervals(transcript_text: str, max_empty_seconds=50):
    async def parse_timecode(timecode):
        """Convert timecode (hh:mm:ss,ms) to milliseconds asynchronously."""
        parts = re.split('[:,]', timecode)
        if len(parts) == 4:
            h, m, s, ms = map(int, parts)
        else:
            raise ValueError(f"Invalid timecode format: {timecode}")
        return (h * 3600000) + (m * 60000) + (s * 1000) + ms

    async def format_timecode(ms):
        """Convert milliseconds to timecode (hh:mm:ss,ms) asynchronously."""
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    async def add_empty_timestamps(transcript, gap_threshold=3000, max_empty_duration=max_empty_seconds * 1000):
        """Insert empty text timestamps when there is a time gap exceeding the threshold asynchronously."""
        lines = transcript.strip().split('\n')
        output = []
        prev_end_time = 0  # Ensure we start from 00:00:00,000
        i = 0
        index = 1

        while i < len(lines):
            if lines[i].isdigit():  # Ensure it's an index line
                i += 1
                if i < len(lines) and '-->' in lines[i]:
                    time_range = lines[i]
                    i += 1
                    text_lines = []
                    while i < len(lines) and not lines[i].isdigit():
                        text_lines.append(lines[i])
                        i += 1
                    text = ' '.join(text_lines)

                    try:
                        start_time, end_time = time_range.split(' --> ')
                        start_ms, end_ms = await parse_timecode(start_time), await parse_timecode(end_time)

                        if start_ms > prev_end_time:
                            empty_start = prev_end_time
                            empty_end = start_ms
                            while empty_end - empty_start > max_empty_duration:
                                empty_chunk_end = empty_start + max_empty_duration
                                output.append(f"{index}\n{await format_timecode(empty_start)} --> {await format_timecode(empty_chunk_end)}\n\n")
                                index += 1
                                empty_start = empty_chunk_end
                            output.append(f"{index}\n{await format_timecode(empty_start)} --> {await format_timecode(empty_end)}\n\n")
                            index += 1

                        output.append(f"{index}\n{time_range}\n{text}\n")
                        index += 1
                        prev_end_time = end_ms
                    except ValueError as e:
                        raise Exception(f"Skipping invalid timestamp: {time_range}, Error: {e}")
            else:
                i += 1  # Skip invalid lines

        return '\n'.join(output)

    return await add_empty_timestamps(transcript_text)


async def format_transcript(transcript: str):
    """
    Formats a transcript by extracting timestamps and text.

    Args:
        transcript (str): Raw transcript content.

    Returns:
        list: List of formatted transcript segments.
    """
    segments = re.findall(
        r"(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n",
        transcript,
        re.DOTALL
    )
    return [f"{timestamp} {text.strip()}" for timestamp, text in segments]


async def calculate_time_differences(strings, seconds_per_frame):
    """
    Calculates time differences between transcript timestamps.

    Args:
        strings (list): List of subtitle timestamps.
        seconds_per_frame (int): Frame extraction interval.

    Returns:
        list: Adjusted time differences for each segment.
    """
    
    time_differences = []
    actual_time_differences = []

    for i, string in enumerate(strings):
        timestamps = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", string)

        if not timestamps:
            continue

        start_timestamp_str, end_timestamp_str = timestamps[0]
        start_timestamp_str = start_timestamp_str.replace(",", ".")
        end_timestamp_str = end_timestamp_str.replace(",", ".")

        start_timestamp = datetime.strptime(start_timestamp_str, "%H:%M:%S.%f")
        end_timestamp = datetime.strptime(end_timestamp_str, "%H:%M:%S.%f")

        time_diff = (end_timestamp - start_timestamp).total_seconds()
        rounded_diff = math.ceil(time_diff / seconds_per_frame) * seconds_per_frame if seconds_per_frame > 1 else round(time_diff)

        actual_time_differences.append(time_diff)
        time_differences.append(rounded_diff)

    return time_differences


async def fetch_frames_based_on_counts(frame_counts, image_frames, seconds_per_frame):
    """
    Fetches frames based on calculated time differences.

    Args:
        frame_counts (list): List of frame count intervals.
        image_frames (list): List of extracted frames.
        seconds_per_frame (int): Interval used for frame extraction.

    Returns:
        list: Frames grouped per transcript section.
    """
    start_index = 0
    frames_per_cluster = []

    for count in frame_counts:
        end_index = start_index + (int(count / seconds_per_frame) if seconds_per_frame > 1 else count)
        frames_per_cluster.append(image_frames[start_index:end_index])
        start_index = end_index

    return frames_per_cluster
