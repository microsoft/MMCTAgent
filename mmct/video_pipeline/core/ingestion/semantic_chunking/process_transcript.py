import re
import numpy as np
import os
from datetime import datetime
import math
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
from mmct.llm_client import LLMClient
# Load environment variables
load_dotenv(find_dotenv(),override=True)

gpt40_client = LLMClient(service_provider=os.getenv("LLM_PROVIDER", "azure"), isAsync=True).get_client()

# Configurable parameters for semantic chunking optimization
OPTIMIZED_SIMILARITY_THRESHOLD = 0.4  # Lower = more content grouped together
SHORT_VIDEO_TIME_LIMIT = 50  # seconds for videos <= 20 minutes
LONG_VIDEO_TIME_LIMIT = 120  # seconds for videos > 20 minutes
VIDEO_DURATION_THRESHOLD = 20  # minutes

async def calculate_transcript_duration(srt_text: str) -> float:
    """
    Calculate the total duration of the transcript in minutes.
    
    Args:
        srt_text (str): The transcript in SRT format
        
    Returns:
        float: Duration in minutes
    """
    try:
        # Find all timestamp ranges
        timestamp_matches = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", srt_text)
        
        if not timestamp_matches:
            return 0.0
        
        # Get the last end timestamp
        last_end_timestamp = timestamp_matches[-1][1]
        
        # Parse the timestamp
        async def parse_timestamp(timestamp):
            h, m, s, ms = map(int, re.split("[:,]", timestamp))
            return h * 3600 + m * 60 + s + ms / 1000
        
        duration_seconds = await parse_timestamp(last_end_timestamp)
        return duration_seconds / 60  # Convert to minutes
        
    except Exception as e:
        logger.warning(f"Could not calculate transcript duration: {e}")
        return 0.0


async def process_transcript(srt_text: str, SIMILARITY_THRESHOLD=None, TIME_LIMIT=None):
    """
    Parses an SRT transcript, removes redundant chunks (e.g., "Music" segments), and performs semantic chunking based on content similarity.

    Parameters:
        srt_text (str): The transcript in SRT format.
        SIMILARITY_THRESHOLD (float): Threshold for cosine similarity to determine chunk separation (default: 0.4).
        TIME_LIMIT (int): Maximum duration (in seconds) of each chunk. If None, calculated based on transcript length.

    Returns:
        str: A formatted string with time ranges and semantically chunked text.
    """
    
    # Use optimized defaults if not provided
    if SIMILARITY_THRESHOLD is None:
        SIMILARITY_THRESHOLD = OPTIMIZED_SIMILARITY_THRESHOLD
    
    # Calculate dynamic time limit if not provided
    if TIME_LIMIT is None:
        transcript_duration_minutes = await calculate_transcript_duration(srt_text)
        if transcript_duration_minutes <= VIDEO_DURATION_THRESHOLD:
            TIME_LIMIT = SHORT_VIDEO_TIME_LIMIT
        else:
            TIME_LIMIT = LONG_VIDEO_TIME_LIMIT
        logger.info(f"Transcript duration: {transcript_duration_minutes:.1f} minutes, using TIME_LIMIT: {TIME_LIMIT}s")
    
    logger.info(f"Using SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}, TIME_LIMIT: {TIME_LIMIT}s")

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
        """
        Groups transcript text into semantic chunks based on chunk-to-sentence similarity.
        Compares accumulated chunk content against new sentences for better semantic coherence.
        """
        if not sentences:
            logger.warning("No sentences provided for semantic chunking")
            return ""

        logger.info(f"🔄 Starting semantic chunking with {len(sentences)} sentences")
        logger.info(f"📊 Configuration: SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}, TIME_LIMIT={TIME_LIMIT}s")

        chunks = {}
        current_chunk = ""
        chunk_start_time = time_stamps[0]
        chunk_end_time = end_times[0]
        embeddings_cache = {}
        
        # Progress tracking
        chunks_created = 0
        embeddings_generated = 0
        cache_hits = 0
        similarity_splits = 0
        time_splits = 0

        for i, sentence in enumerate(sentences):
            # Progress logging every 50 sentences
            if i > 0 and i % 50 == 0:
                progress_pct = (i / len(sentences)) * 100
                logger.info(f"⏳ Processing progress: {i}/{len(sentences)} sentences ({progress_pct:.1f}%) | Chunks created: {chunks_created}")

            # Get embedding for current sentence
            if sentence in embeddings_cache:
                sentence_embedding = embeddings_cache[sentence]
                cache_hits += 1
            else:
                sentence_embedding = await create_embedding(sentence)
                if sentence_embedding is None:
                    logger.warning(f"Failed to create embedding for sentence {i}, skipping")
                    continue  # Skip if embedding fails
                embeddings_cache[sentence] = sentence_embedding
                embeddings_generated += 1

            if i == 0:
                # Initialize with first sentence
                current_chunk = sentence + " "
                chunk_start_time = time_stamps[i]
                chunk_end_time = end_times[i]
                logger.info(f"🎯 Initialized first chunk at {await format_timestamp(chunk_start_time)}")
                continue

            # Get embedding for current chunk content
            # Use chunk position + content hash for unique cache key to avoid collisions
            chunk_text = current_chunk.strip()
            chunk_embedding_key = f"chunk_{chunks_created}_{hash(chunk_text)}"
            
            if chunk_embedding_key in embeddings_cache:
                chunk_embedding = embeddings_cache[chunk_embedding_key]
                cache_hits += 1
            else:
                chunk_embedding = await create_embedding(chunk_text)
                if chunk_embedding is None:
                    logger.warning(f"Failed to create chunk embedding at sentence {i}, adding sentence to chunk anyway")
                    # Don't skip - add sentence to chunk even if embedding fails
                    current_chunk += sentence + " "
                    chunk_end_time = end_times[i]
                    continue
                embeddings_cache[chunk_embedding_key] = chunk_embedding
                embeddings_generated += 1

            # Calculate similarity between current chunk and new sentence
            similarity = await calculate_cosine_similarity(chunk_embedding, sentence_embedding)
            
            # Check if we should start a new chunk
            time_exceeded = (end_times[i] - chunk_start_time) > TIME_LIMIT
            similarity_too_low = similarity < SIMILARITY_THRESHOLD
            
            if similarity_too_low or time_exceeded:
                # Finalize current chunk
                if current_chunk.strip():
                    chunk_duration = chunk_end_time - chunk_start_time
                    chunks[f"{await format_timestamp(chunk_start_time)} --> {await format_timestamp(chunk_end_time)}"] = current_chunk.strip()
                    chunks_created += 1
                    
                    # High-level chunk creation logging
                    reason = "LOW_SIMILARITY" if similarity_too_low else "TIME_LIMIT"
                    word_count = len(current_chunk.strip().split())
                    logger.info(f"📦 Chunk #{chunks_created} created: {chunk_duration:.1f}s duration, {word_count} words | Reason: {reason} (sim={similarity:.3f})")
                    
                    # Track split reasons
                    if similarity_too_low:
                        similarity_splits += 1
                    if time_exceeded:
                        time_splits += 1
                
                # Start new chunk
                current_chunk = sentence + " "
                chunk_start_time = time_stamps[i]
                chunk_end_time = end_times[i]
            else:
                # Add sentence to current chunk
                current_chunk += sentence + " "
                chunk_end_time = end_times[i]

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunk_duration = chunk_end_time - chunk_start_time
            chunks[f"{await format_timestamp(chunk_start_time)} --> {await format_timestamp(chunk_end_time)}"] = current_chunk.strip()
            chunks_created += 1
            word_count = len(current_chunk.strip().split())
            logger.info(f"📦 Final Chunk #{chunks_created} created: {chunk_duration:.1f}s duration, {word_count} words | Reason: END_OF_TRANSCRIPT")

        # Final summary logging
        reduction_pct = ((len(sentences) - len(chunks)) / len(sentences)) * 100 if len(sentences) > 0 else 0
        logger.info(f"✅ Semantic chunking complete!")
        logger.info(f"📈 Results: {len(sentences)} sentences → {len(chunks)} chunks ({reduction_pct:.1f}% reduction)")
        logger.info(f"🧠 Embeddings: {embeddings_generated} generated, {cache_hits} cache hits")
        logger.info(f"🔀 Split reasons: {similarity_splits} similarity, {time_splits} time limit")
        
        return "\n\n".join(f"{idx+1}\n{time_range}\n{chunk}" for idx, (time_range, chunk) in enumerate(chunks.items()))

    sentences, time_stamps, end_times = await parse_transcript(srt_text)
    logger.info(f"Parsed {len(sentences)} sentences from transcript")
    
    result = await semantic_chunking(sentences, time_stamps, end_times)
    return result

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


async def parse_timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp (HH:MM:SS,mmm) to seconds.
    
    Args:
        timestamp (str): Timestamp in format HH:MM:SS,mmm
        
    Returns:
        float: Timestamp in seconds
    """
    h, m, s, ms = map(int, re.split("[:,]", timestamp))
    return h * 3600 + m * 60 + s + ms / 1000


async def merge_short_clusters(clusters: list, min_duration_seconds: int = 30) -> list:
    """
    Merge clusters shorter than min_duration with subsequent clusters to improve chapter quality.
    
    Args:
        clusters (list): List of cluster strings with format "HH:MM:SS,mmm --> HH:MM:SS,mmm text"
        min_duration_seconds (int): Minimum duration threshold for clusters
        
    Returns:
        list: List of merged clusters
    """
    if not clusters:
        return clusters
        
    async def extract_cluster_info(cluster_text: str):
        """Extract duration, start time, end time, and text from cluster."""
        timestamps = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", cluster_text)
        if timestamps:
            start, end = timestamps[0]
            start_sec = await parse_timestamp_to_seconds(start)
            end_sec = await parse_timestamp_to_seconds(end)
            duration = end_sec - start_sec
            
            # Extract text content (everything after the timestamp)
            text_match = re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s+(.*)", cluster_text)
            text = text_match.group(1).strip() if text_match else ""
            
            return duration, start, end, text
        return 0, None, None, ""
    
    async def format_timestamp_from_seconds(seconds: float) -> str:
        """Convert seconds back to timestamp format."""
        ms = int((seconds % 1) * 1000)
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    merged_clusters = []
    i = 0
    
    logger.info(f"Starting cluster merging. Original clusters: {len(clusters)}, Min duration: {min_duration_seconds}s")
    
    while i < len(clusters):
        current_cluster = clusters[i]
        duration, start_time, end_time, text = await extract_cluster_info(current_cluster)
        
        # If cluster is too short and we have a next cluster, try to merge
        if duration < min_duration_seconds and i + 1 < len(clusters):
            next_cluster = clusters[i + 1]
            next_duration, next_start, next_end, next_text = await extract_cluster_info(next_cluster)
            
            # Create merged cluster with combined text and extended time range
            if start_time and next_end:
                merged_text = f"{start_time} --> {next_end} {text} {next_text}".strip()
                merged_clusters.append(merged_text)
                logger.debug(f"Merged cluster: {duration:.1f}s + {next_duration:.1f}s = {duration + next_duration:.1f}s")
                i += 2  # Skip next cluster since we merged it
            else:
                # If timestamp parsing failed, keep original
                merged_clusters.append(current_cluster)
                i += 1
        else:
            # Cluster is long enough or it's the last one, keep as is
            merged_clusters.append(current_cluster)
            i += 1
    
    logger.info(f"Cluster merging complete. Final clusters: {len(merged_clusters)} (reduced by {len(clusters) - len(merged_clusters)})")
    return merged_clusters
