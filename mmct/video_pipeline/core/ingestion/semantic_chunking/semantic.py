import warnings
import uuid
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Union, Optional
from azure.search.documents import SearchClient
from mmct.video_pipeline.utils.ai_search_client import AISearchClient, AISearchDocument
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from mmct.video_pipeline.core.ingestion.semantic_chunking.process_transcript import (
    process_transcript,
    add_empty_intervals,
    format_transcript,
    calculate_time_differences,
    fetch_frames_based_on_counts,
    merge_short_clusters
)
from loguru import logger
from mmct.video_pipeline.core.ingestion.chapter_generator.generate_chapter import ChapterGeneration
from mmct.llm_client import LLMClient

from dotenv import load_dotenv, find_dotenv
# Load environment variables
load_dotenv(find_dotenv(),override=True)


class SemanticChunking:
    def __init__(self, hash_id:str, index_name:str, transcript:str,blob_urls,base64Frames, frame_stacking_grid_size: int = 4)->None:
        if os.environ.get("MANAGED_IDENTITY", None) is None:
            raise Exception(
                "MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
            )
        self.azure_managed_identity = os.environ.get(
            "MANAGED_IDENTITY", ""
        ).upper()
        if self.azure_managed_identity == "TRUE":
            logger.info("auth via managed indentity!")
            # Use Azure CLI credential if available, fallback to DefaultAzureCredential
            self.token_provider = self._get_credential()

        else:
            key = os.getenv("SEARCH_SERVICE_KEY",None)
            if key is None:
                raise Exception("Key is missing for Azure AI Search!")
            self.token_provider = AzureKeyCredential(key)
            
        self.chapter_responses = []
        self.chapter_response_strings = []
        self.chapter_transcripts = []
        self.clusters = []
        self.frames_per_cluster = []
        self.subject_data = {'subject': 'None', 'variety_of_subject': 'None'}
        self.upload_doc = []
        self.transcript = transcript
        self.base64Frames = base64Frames
        self.hash_id = hash_id
        self.index_name = index_name
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.chapter_generator = ChapterGeneration(frame_stacking_grid_size=frame_stacking_grid_size)
        self.embed_client = LLMClient(service_provider=os.getenv("LLM_PROVIDER", "azure"), isAsync=True, embedding=True).get_client()
        self.index_client = AISearchClient(
            endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=self.index_name,
            credential=self.token_provider
        )
        self.blob_urls = blob_urls
        #self.index_client = SearchClient(endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"), index_name=self.index_name, credential=self.token_provider)
    async def _create_search_index(self):
        created = await self.index_client.check_and_create_index()
        if created:
            logger.info(f"Index {self.index_name} created successfully.")
        else:
            logger.info(f"Index {self.index_name} already exists.")
    
    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            from azure.identity import AzureCliCredential
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            cli_credential.get_token("https://search.azure.com/.default")
            return cli_credential
        except Exception:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
    
    async def _create_embedding_normal(self,text:str) -> List[float]:
        try:
            response = await self.embed_client.embeddings.create(
                input=[text], 
                model= os.getenv("EMBEDDING_SERVICE_MODEL_NAME" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_EMBEDDING_MODEL_NAME")
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f'Failed to create embedding:{e}',)

    async def _pre_process(self):
        # 1. Check duplicate
        existing = await self.index_client.check_if_exists(hash_id=self.hash_id)
        if existing:
            return True
        
        logger.info(f"Original transcript before processing: {self.transcript[:500]}...")
        self.transcript = await process_transcript(srt_text=self.transcript) #Parses an SRT transcript, removes redundant chunks (e.g., "Music" segments), and performs semantic chunking based on content similarity.
        logger.info(f"Transcript after process_transcript: {self.transcript[:500]}...")
        self.transcript = await add_empty_intervals(transcript_text=self.transcript)
        
        titled_transcript = "\nVideo Transcript: " + self.transcript
        logger.info(f"Titled transcript:{self.transcript}")
        self.clusters = await format_transcript(transcript=self.transcript)
        logger.info(f"Original clusters count: {len(self.clusters)}")
        
        # Merge short duration clusters to improve chapter quality and reduce processing time
        self.clusters = await merge_short_clusters(self.clusters, min_duration_seconds=30)
        logger.info(f"Clusters after merging: {len(self.clusters)}")
        
        if not self.clusters:
            warnings.warn("Formatted Transcript is Empty.", RuntimeWarning)
            logger.error("No clusters generated from transcript - this will result in no documents to index!")
            # Initialize empty attributes to prevent AttributeError
            self.frames_per_cluster = []
            self.subject_data = {'subject': 'None', 'variety_of_subject': 'None'}
            return False
        time_differences = await calculate_time_differences(self.clusters,1)
        
        self.frames_per_cluster = await fetch_frames_based_on_counts(time_differences, self.base64Frames, 1)
        
        self.subject_data = await (self.chapter_generator.subject_and_variety(transcript=titled_transcript))
        self.subject_data = eval(self.subject_data)
        logger.info(self.subject_data)
        return False
        
        
    async def _create_chapters(self):
        if not self.clusters or not self.frames_per_cluster:
            logger.warning("No clusters or frames available for chapter creation")
            return
        
        # Create semaphore to limit concurrent Azure OpenAI requests
        # Adjust this value based on your Azure OpenAI quota and rate limits
        max_concurrent_requests = 3  # Conservative limit to avoid rate limiting
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async def create_single_chapter(idx, seg, fr):
            """Create a single chapter with retry logic and rate limiting."""
            async with semaphore:  # Limit concurrent requests
                attempts = 0
                max_attempts = 3
                delay = 1
                while attempts < max_attempts:
                    try:
                        # Get ChapterCreationResponse instance
                        chapter_response = await self.chapter_generator.Chapters_creation(
                            transcript = seg, frames = fr, categories = "", subject_variety = self.subject_data
                        )
                        logger.info(f"Chapter {idx}: transcript segment:{seg}")
                        logger.info(f"Chapter {idx}: raw chapter:{chapter_response}")
                        logger.info(f"Chapter {idx}: string chapter:{chapter_response.__str__(transcript=seg)}")
                        
                        if chapter_response is not None:
                            # Create string representation with transcript only for indexing
                            chapter_response_str = chapter_response.__str__(transcript=seg)
                            return idx, chapter_response, seg, chapter_response_str
                        else:
                            logger.warning(f"Chapter {idx}: No response received, attempting retry {attempts + 1}/{max_attempts}")
                            attempts += 1
                            if attempts < max_attempts:
                                await asyncio.sleep(delay)
                                delay *= 2
                            continue
                            
                        break
                    except Exception as e:
                        # Check if it's a rate limiting error
                        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                            logger.warning(f"Chapter {idx}: Rate limit hit, waiting longer before retry...")
                            await asyncio.sleep(delay * 2)  # Wait longer for rate limit errors
                        else:
                            logger.error(f"Chapter {idx}: Error on attempt {attempts + 1}: {e}")
                        
                        attempts += 1
                        if attempts < max_attempts:
                            await asyncio.sleep(delay)
                            delay *= 2
                        else:
                            logger.error(f"Chapter {idx}: Failed after {max_attempts} attempts")
                            raise
                
                return None  # Failed after all attempts
        
        # Create tasks for all chapters to process with controlled concurrency
        logger.info(f"Creating {len(self.clusters)} chapters with max {max_concurrent_requests} concurrent requests...")
        tasks = []
        for idx, (seg, fr) in enumerate(zip(self.clusters, self.frames_per_cluster)):
            task = create_single_chapter(idx, seg, fr)
            tasks.append(task)
        
        # Execute all chapter creation tasks with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results in order to maintain chapter sequence
        successful_chapters = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chapter creation failed with exception: {result}")
                continue
            elif result is not None:
                successful_chapters.append(result)
        
        # Sort by chapter index to maintain order
        successful_chapters.sort(key=lambda x: x[0])
        
        # Store results in class attributes
        for idx, chapter_response, seg, chapter_response_str in successful_chapters:
            # Store the ChapterCreationResponse object for summary extraction
            self.chapter_responses.append(chapter_response)
            
            # Store the transcript segment
            self.chapter_transcripts.append(seg)
            
            # Store string representation for indexing
            self.chapter_response_strings.append(chapter_response_str)
        
        logger.info(f"Chapter Generation Completed! Successfully created {len(self.chapter_responses)} chapters in parallel.")
        
        
    async def _ingest(self, video_blob_url, youtube_url=None):
        url_info = f'BLOB={video_blob_url}\nYT_URL={youtube_url}'
        doc_objects: List[AISearchDocument] = [] 
        current_time = datetime.now()
        logger.info(f"Creating documents from {len(self.chapter_responses)} chapters")
        for chapter_response, chapter_transcript in zip(self.chapter_responses, self.chapter_transcripts):
            chapter_content_str = chapter_response.__str__(transcript=chapter_transcript)
            obj = AISearchDocument(
                id=str(uuid.uuid4()),
                hash_video_id=self.hash_id,
                topic_of_video=chapter_response.Topic_of_video or "None",
                action_taken=chapter_response.Action_taken or "None",
                detailed_summary=chapter_response.Detailed_summary or "None",
                category=chapter_response.Category or "None",
                sub_category=chapter_response.Sub_category or "None",
                text_from_scene=chapter_response.Text_from_scene or "None",
                youtube_url=youtube_url or "None",
                time=current_time,
                chapter_transcript=chapter_transcript,
                subject=self.subject_data['subject'] or "None",
                variety=self.subject_data['variety_of_subject'] or "None",
                blob_audio_url=self.blob_urls['audio_blob_url'].split(".net")[-1][1:] or "None",
                blob_video_url=video_blob_url.split(".net")[-1][1:] or "None",
                blob_transcript_file_url=self.blob_urls['transcript_blob_url'].split(".net")[-1][1:] or "None",
                blob_frames_folder_path=self.blob_urls['frames_blob_folder_url'].split(".net")[-1][1:] or "None",
                blob_timestamps_file_url=self.blob_urls['timestamps_blob_url'].split(".net")[-1][1:] or "None",
                blob_transcript_and_summary_file_url=self.blob_urls['transcript_and_summary_file_url'].split(".net")[-1][1:] or "None",
                embeddings=await self._create_embedding_normal(chapter_content_str)
            )
            doc_objects.append(obj)
            
        logger.info(f"Generated {len(doc_objects)} documents to upload")
        if not doc_objects:
            logger.error("No documents created - cannot upload to search index!")
            return
            
        await self.index_client.upload_documents(documents=[doc.model_dump() for doc in doc_objects])
        
    async def run(self,video_blob_url, youtube_url=None):
        await self._create_search_index() # checking if index is available, if not then creating the same.
        is_exist = await self._pre_process()
        if is_exist:
            logger.info("Document already exists in the index.")
            return None, None, is_exist
        await self._create_chapters()
        await self._ingest(video_blob_url=video_blob_url, youtube_url=youtube_url)
        await self.index_client.close()
        return self.chapter_responses, self.chapter_transcripts, is_exist
        
if __name__=="__main__":
    semantic_chunker = SemanticChunking()
    asyncio.run(semantic_chunker.run())