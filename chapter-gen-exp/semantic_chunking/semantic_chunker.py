"""
Semantic Chunker Module

This module handles semantic clustering of video transcripts.
It processes transcripts and groups semantically similar segments together.
"""

import re
import warnings
import asyncio
import numpy as np
import os
from typing import List
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
from mmct.providers.factory import provider_factory
from mmct.video_pipeline.core.ingestion.semantic_chunking.process_transcript import TranscriptSegment

# Load environment variables
load_dotenv(find_dotenv(), override=True)


class SemanticChunker:
    """
    Handles semantic clustering of video transcripts.

    This class is responsible ONLY for semantic chunking operations:
    - Parsing transcripts
    - Performing semantic clustering
    - Adding empty intervals for gaps
    - Merging short clusters
    """

    # Configurable parameters for semantic chunking optimization
    OPTIMIZED_SIMILARITY_THRESHOLD = 0.4  # Lower = more content grouped together
    SHORT_VIDEO_TIME_LIMIT = 50  # seconds for videos <= 20 minutes
    LONG_VIDEO_TIME_LIMIT = 120  # seconds for videos > 20 minutes
    VIDEO_DURATION_THRESHOLD = 20  # minutes

    def __init__(self, transcript: str):
        """
        Initialize SemanticChunker.

        Args:
            transcript (str): Raw SRT transcript text to be processed
        """
        self.transcript = transcript
        self.chunked_segments = []
        self.embedding_provider = provider_factory.create_embedding_provider()

    async def _calculate_transcript_duration(self, srt_text: str) -> float:
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

    async def _parse_transcript(self, srt_text: str) -> List[TranscriptSegment]:
        """
        Extracts sentences and timestamps from SRT text and returns as Pydantic objects.

        Args:
            srt_text (str): The transcript in SRT format

        Returns:
            List[TranscriptSegment]: List of parsed transcript segments with sentence, start_time, and end_time
        """
        async def parse_timestamp(timestamp):
            h, m, s, ms = map(int, re.split("[:,]", timestamp))
            return h * 3600 + m * 60 + s + ms / 1000

        entries = re.findall(
            r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\d+\n|\Z)",
            srt_text,
            re.DOTALL
        )
        segments = []

        for index, start_time, end_time, text in entries:
            text = text.replace("\n", " ").strip()
            if text.lower() != "music" and text:
                segment = TranscriptSegment(
                    sentence=text,
                    start_time=await parse_timestamp(start_time),
                    end_time=await parse_timestamp(end_time)
                )
                segments.append(segment)

        return segments

    async def _create_batch_embeddings(self, texts, batch_size=100):
        """Generates embeddings for multiple texts in batches using Azure Embeddings API."""
        logger.info(f"🔄 Creating batch embeddings for {len(texts)} texts")
        embeddings = await self.embedding_provider.batch_embedding(texts)
        logger.info(f"✅ Batch embedding complete: {len(embeddings)} embeddings created")
        return embeddings

    async def _calculate_cosine_similarity(self, vec1, vec2):
        """Computes cosine similarity between two vectors asynchronously."""
        return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]

    async def _calculate_chunk_centroid(self, embeddings):
        """Calculate the centroid (mean) of a list of embeddings."""
        if not embeddings:
            return None
        embeddings_array = np.array(embeddings)
        return np.mean(embeddings_array, axis=0)

    async def _format_timestamp(self, seconds):
        """Format seconds to SRT timestamp format."""
        ms = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d},{ms:03d}"

    async def _perform_semantic_chunking(self, sentences, time_stamps, end_times, SIMILARITY_THRESHOLD, TIME_LIMIT):
        """
        Groups transcript text into semantic chunks using batch embeddings and greedy centroid-based approach.
        Compares new sentences against chunk centroid for better semantic coherence.
        """
        if not sentences:
            logger.warning("No sentences provided for semantic chunking")
            return {}

        logger.info(f"🔄 Starting semantic chunking with {len(sentences)} sentences")
        logger.info(f"📊 Configuration: SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}, TIME_LIMIT={TIME_LIMIT}s")

        # Step 1: Batch embed all sentences
        logger.info("🧠 Creating batch embeddings for all sentences...")
        sentence_embeddings = await self._create_batch_embeddings(sentences)

        # Filter out None embeddings and corresponding data
        valid_data = []
        for i, embedding in enumerate(sentence_embeddings):
            if embedding is not None:
                valid_data.append((sentences[i], time_stamps[i], end_times[i], embedding))
            else:
                logger.warning(f"Skipping sentence {i} due to failed embedding")

        if not valid_data:
            logger.error("No valid embeddings created")
            return {}

        logger.info(f"✅ Created {len(valid_data)} valid embeddings")

        # Step 2: Greedy chunking with centroid comparison
        chunks = {}
        chunks_created = 0
        similarity_splits = 0
        time_splits = 0

        # Initialize first chunk
        current_chunk_sentences = [valid_data[0][0]]
        current_chunk_embeddings = [valid_data[0][3]]
        chunk_start_time = valid_data[0][1]
        chunk_end_time = valid_data[0][2]

        logger.info(f"🎯 Initialized first chunk at {await self._format_timestamp(chunk_start_time)}")

        for i in range(1, len(valid_data)):
            sentence, start_time, end_time, sentence_embedding = valid_data[i]

            # Progress logging every 50 sentences
            if i % 50 == 0:
                progress_pct = (i / len(valid_data)) * 100
                logger.info(f"⏳ Processing progress: {i}/{len(valid_data)} sentences ({progress_pct:.1f}%) | Chunks created: {chunks_created}")

            # Calculate current chunk centroid
            chunk_centroid = await self._calculate_chunk_centroid(current_chunk_embeddings)
            if chunk_centroid is None:
                logger.warning(f"Failed to calculate centroid at sentence {i}, adding to chunk anyway")
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(sentence_embedding)
                chunk_end_time = end_time
                continue

            # Calculate similarity between sentence and chunk centroid
            similarity = await self._calculate_cosine_similarity(chunk_centroid, sentence_embedding)

            # Check if we should start a new chunk
            time_exceeded = (end_time - chunk_start_time) > TIME_LIMIT
            similarity_too_low = similarity < SIMILARITY_THRESHOLD

            if similarity_too_low or time_exceeded:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_duration = chunk_end_time - chunk_start_time
                chunks[f"{await self._format_timestamp(chunk_start_time)} --> {await self._format_timestamp(chunk_end_time)}"] = chunk_text
                chunks_created += 1

                # Logging
                reason = "LOW_SIMILARITY" if similarity_too_low else "TIME_LIMIT"
                word_count = len(chunk_text.split())
                logger.info(f"📦 Chunk #{chunks_created} created: {chunk_duration:.1f}s duration, {word_count} words | Reason: {reason} (sim={similarity:.3f})")

                # Track split reasons
                if similarity_too_low:
                    similarity_splits += 1
                if time_exceeded:
                    time_splits += 1

                # Start new chunk with current sentence
                current_chunk_sentences = [sentence]
                current_chunk_embeddings = [sentence_embedding]
                chunk_start_time = start_time
                chunk_end_time = end_time
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(sentence_embedding)
                chunk_end_time = end_time

        # Add the last chunk if it exists
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_duration = chunk_end_time - chunk_start_time
            chunks[f"{await self._format_timestamp(chunk_start_time)} --> {await self._format_timestamp(chunk_end_time)}"] = chunk_text
            chunks_created += 1
            word_count = len(chunk_text.split())
            logger.info(f"📦 Final Chunk #{chunks_created} created: {chunk_duration:.1f}s duration, {word_count} words | Reason: END_OF_TRANSCRIPT")

        # Final summary logging
        reduction_pct = ((len(valid_data) - len(chunks)) / len(valid_data)) * 100 if len(valid_data) > 0 else 0
        logger.info(f"✅ Semantic chunking complete!")
        logger.info(f"📈 Results: {len(valid_data)} sentences → {len(chunks)} chunks ({reduction_pct:.1f}% reduction)")
        logger.info(f"🔀 Split reasons: {similarity_splits} similarity, {time_splits} time limit")

        return chunks

    async def _semantic_chunking(self, srt_text: str, SIMILARITY_THRESHOLD=None, TIME_LIMIT=None) -> List[TranscriptSegment]:
        """
        Performs semantic chunking on parsed transcript segments based on content similarity.

        Parameters:
            srt_text (str): The transcript in SRT format.
            SIMILARITY_THRESHOLD (float): Threshold for cosine similarity to determine chunk separation (default: 0.4).
            TIME_LIMIT (int): Maximum duration (in seconds) of each chunk. If None, calculated based on transcript length.

        Returns:
            List[TranscriptSegment]: List of semantically chunked segments with sentence, start_time, and end_time.
        """

        # Use optimized defaults if not provided
        if SIMILARITY_THRESHOLD is None:
            SIMILARITY_THRESHOLD = self.OPTIMIZED_SIMILARITY_THRESHOLD

        # Calculate dynamic time limit if not provided
        if TIME_LIMIT is None:
            transcript_duration_minutes = await self._calculate_transcript_duration(srt_text)
            if transcript_duration_minutes <= self.VIDEO_DURATION_THRESHOLD:
                TIME_LIMIT = self.SHORT_VIDEO_TIME_LIMIT
            else:
                TIME_LIMIT = self.LONG_VIDEO_TIME_LIMIT
            logger.info(f"Transcript duration: {transcript_duration_minutes:.1f} minutes, using TIME_LIMIT: {TIME_LIMIT}s")

        logger.info(f"Using SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}, TIME_LIMIT: {TIME_LIMIT}s")

        # Parse transcript into segments
        segments = await self._parse_transcript(srt_text)
        logger.info(f"Parsed {len(segments)} sentences from transcript")

        # Extract data from segments
        sentences = [seg.sentence for seg in segments]
        time_stamps = [seg.start_time for seg in segments]
        end_times = [seg.end_time for seg in segments]

        # Perform semantic chunking and get the chunks dict
        chunks_dict = await self._perform_semantic_chunking(sentences, time_stamps, end_times, SIMILARITY_THRESHOLD, TIME_LIMIT)

        # Convert chunks dict to List[TranscriptSegment]
        chunked_segments = []
        for time_range, chunk_text in chunks_dict.items():
            # Parse time range: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
            start_str, end_str = time_range.split(" --> ")

            async def parse_timestamp_str(timestamp):
                h, m, s, ms = map(int, re.split("[:,]", timestamp))
                return h * 3600 + m * 60 + s + ms / 1000

            start_time = await parse_timestamp_str(start_str)
            end_time = await parse_timestamp_str(end_str)

            segment = TranscriptSegment(
                sentence=chunk_text,
                start_time=start_time,
                end_time=end_time
            )
            chunked_segments.append(segment)

        logger.info(f"Created {len(chunked_segments)} semantic chunks")
        return chunked_segments

    async def _add_empty_intervals(self, segments: List[TranscriptSegment], max_empty_seconds=50) -> List[TranscriptSegment]:
        """
        Insert empty intervals between segments when there are gaps.

        Args:
            segments (List[TranscriptSegment]): List of transcript segments
            max_empty_seconds (int): Maximum duration for a single empty interval

        Returns:
            List[TranscriptSegment]: Segments with empty intervals added
        """
        if not segments:
            return segments

        result = []
        prev_end_time = 0.0

        for segment in segments:
            # Check if there's a gap between previous segment and current
            if segment.start_time > prev_end_time:
                gap_duration = segment.start_time - prev_end_time

                # Split large gaps into chunks
                if gap_duration > max_empty_seconds:
                    empty_start = prev_end_time
                    while (segment.start_time - empty_start) > max_empty_seconds:
                        empty_segment = TranscriptSegment(
                            sentence="",
                            start_time=empty_start,
                            end_time=empty_start + max_empty_seconds
                        )
                        result.append(empty_segment)
                        empty_start += max_empty_seconds

                    # Add final chunk to reach the segment start
                    if empty_start < segment.start_time:
                        empty_segment = TranscriptSegment(
                            sentence="",
                            start_time=empty_start,
                            end_time=segment.start_time
                        )
                        result.append(empty_segment)
                else:
                    # Add single empty interval
                    empty_segment = TranscriptSegment(
                        sentence="",
                        start_time=prev_end_time,
                        end_time=segment.start_time
                    )
                    result.append(empty_segment)

            # Add the actual segment
            result.append(segment)
            prev_end_time = segment.end_time

        logger.info(f"Added empty intervals: {len(segments)} segments → {len(result)} segments (added {len(result) - len(segments)} empty intervals)")
        return result

    async def _merge_short_clusters(self, clusters: List[TranscriptSegment], min_duration_seconds: int = 30) -> List[TranscriptSegment]:
        """
        Merge clusters shorter than min_duration with subsequent clusters to improve chapter quality.

        Args:
            clusters (List[TranscriptSegment]): List of TranscriptSegment objects
            min_duration_seconds (int): Minimum duration threshold for clusters

        Returns:
            List[TranscriptSegment]: List of merged TranscriptSegment objects
        """
        if not clusters:
            return clusters

        merged_clusters = []
        i = 0

        logger.info(f"Starting cluster merging. Original clusters: {len(clusters)}, Min duration: {min_duration_seconds}s")

        while i < len(clusters):
            current_segment = clusters[i]
            duration = current_segment.end_time - current_segment.start_time

            # If cluster is too short and we have a next cluster, try to merge
            if duration < min_duration_seconds and i + 1 < len(clusters):
                next_segment = clusters[i + 1]
                next_duration = next_segment.end_time - next_segment.start_time

                # Create merged segment with combined text and extended time range
                merged_segment = TranscriptSegment(
                    sentence=f"{current_segment.sentence} {next_segment.sentence}",
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time
                )
                merged_clusters.append(merged_segment)
                logger.debug(f"Merged cluster: {duration:.1f}s + {next_duration:.1f}s = {duration + next_duration:.1f}s")
                i += 2  # Skip next cluster since we merged it
            else:
                # Cluster is long enough or it's the last one, keep as is
                merged_clusters.append(current_segment)
                i += 1

        logger.info(f"Cluster merging complete. Final clusters: {len(merged_clusters)} (reduced by {len(clusters) - len(merged_clusters)})")
        return merged_clusters

    async def _process_and_chunk_transcript(self, srt_text: str) -> List:
        """
        Process the transcript through semantic chunking pipeline.

        Args:
            srt_text (str): Raw SRT transcript text

        Returns:
            List[TranscriptSegment]: Processed and chunked transcript segments
        """
        # Perform semantic chunking - returns List[TranscriptSegment]
        chunked_segments = await self._semantic_chunking(srt_text=srt_text)
        logger.info(f"Semantic chunking complete: {len(chunked_segments)} chunks created")

        # Add empty intervals for gaps in the transcript
        chunked_segments = await self._add_empty_intervals(chunked_segments)
        logger.info(f"After adding empty intervals: {len(chunked_segments)} segments")

        # Merge short duration clusters to improve chapter quality and reduce processing time
        chunked_segments = await self._merge_short_clusters(chunked_segments, min_duration_seconds=30)
        logger.info(f"Clusters after merging: {len(chunked_segments)}")

        return chunked_segments

    async def run(self) -> List:
        """
        Process the transcript through the complete semantic chunking pipeline.

        Returns:
            List[TranscriptSegment]: Semantically chunked transcript segments, or empty list on error
        """
        logger.info(f"Starting semantic chunking for transcript: {self.transcript[:500]}...")

        # Process through semantic chunking pipeline
        chunked_segments = await self._process_and_chunk_transcript(srt_text=self.transcript)

        # Validate chunks
        if not chunked_segments:
            warnings.warn("Formatted Transcript is Empty.", RuntimeWarning)
            logger.error("No clusters generated from transcript!")
            return []

        # Store the chunked segments
        self.chunked_segments = chunked_segments
        logger.info(f"Semantic chunking completed: {len(chunked_segments)} final segments")

        return self.chunked_segments


if __name__ == "__main__":
    # Example usage
    sample_transcript = """1
00:00:00,000 --> 00:00:05,000
This is a sample transcript.

2
00:00:05,000 --> 00:00:10,000
It contains multiple segments."""

    chunker = SemanticChunker(transcript=sample_transcript)
    asyncio.run(chunker.run())
