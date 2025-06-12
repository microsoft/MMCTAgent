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
    fetch_frames_based_on_counts
)
from loguru import logger
from mmct.video_pipeline.core.ingestion.chapter_generator.generate_chapter import ChapterGeneration
from mmct.llm_client import LLMClient

from dotenv import load_dotenv, find_dotenv
# Load environment variables
load_dotenv(find_dotenv(),override=True)


class SemanticChunking:
    def __init__(self, hash_id:str, index_name:str, transcript:str,blob_urls,base64Frames)->None:
        if os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY", None) is None:
            raise Exception(
                "AZURE_OPENAI_MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
            )
        self.azure_managed_identity = os.environ.get(
            "AZURE_OPENAI_MANAGED_IDENTITY", ""
        ).upper()
        if self.azure_managed_identity == "TRUE":
            logger.info("auth via managed indentity!")
            self.token_provider = DefaultAzureCredential()

        else:
            key = os.getenv("AZURE_AI_SEARCH_KEY",None)
            if key is None:
                raise Exception("Key is missing for Azure AI Search!")
            self.token_provider = AzureKeyCredential(key)
            
        self.chapter_responses = []
        self.chapter_response_strings = []
        self.chapter_transcripts = []
        self.upload_doc = []
        self.transcript = transcript
        self.base64Frames = base64Frames
        self.hash_id = hash_id
        self.index_name = index_name
        self.chapter_generator = ChapterGeneration()
        self.embed_client = LLMClient(service_provider=os.getenv("LLM_PROVIDER", "azure"), isAsync=True, embedding=True).get_client()
        self.index_client = AISearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            index_name=self.index_name,
            credential=self.token_provider
        )
        self.blob_urls = blob_urls
        #self.index_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), index_name=self.index_name, credential=self.token_provider)
    async def _create_search_index(self):
        created = await self.index_client.check_and_create_index()
        if created:
            logger.info(f"Index {self.index_name} created successfully.")
        else:
            logger.info(f"Index {self.index_name} already exists.")
    
    async def _create_embedding_normal(self,text:str) -> List[float]:
        try:
            response = await self.embed_client.embeddings.create(
                input=[text], 
                model= os.getenv("AZURE_EMBEDDING_MODEL" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_EMBEDDING_MODEL")
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f'Failed to create embedding:{e}',)

    async def _pre_process(self):
        # 1. Check duplicate
        existing = await self.index_client.check_if_exists(hash_id=self.hash_id)
        if existing:
            return True
        
        self.transcript = await process_transcript(srt_text=self.transcript) #Parses an SRT transcript, removes redundant chunks (e.g., "Music" segments), and performs semantic chunking based on content similarity.
        self.transcript = await add_empty_intervals(transcript_text=self.transcript)
        
        titled_transcript = "\nVideo Transcript: " + self.transcript
        logger.info(f"Titled transcript:{self.transcript}")
        self.clusters = await format_transcript(transcript=self.transcript)
        logger.info(f"Clusters:{self.clusters}")
        if not self.clusters:
            warnings.warn("Formatted Transcript is Empty.", RuntimeWarning)
        time_differences = await calculate_time_differences(self.clusters,1)
        
        self.frames_per_cluster = await fetch_frames_based_on_counts(time_differences, self.base64Frames, 1)
        
        self.species_data = await (self.chapter_generator.species_and_variety(transcript=titled_transcript))
        self.species_data = eval(self.species_data)
        logger.info(self.species_data)
        return False
        
        
    async def _create_chapters(self):
        for idx, (seg, fr) in enumerate(zip(self.clusters, self.frames_per_cluster)):
            attempts = 0
            max_attempts = 3
            delay = 1
            while attempts < max_attempts:
                try:
                    # Get ChapterCreationResponse instance
                    chapter_response = await self.chapter_generator.Chapters_creation(
                        transcript = seg, frames = fr, categories = "", species_variety = self.species_data
                    )
                    logger.info(f"transcript segment:{seg}")
                    logger.info(f"raw chapter:{chapter_response}")
                    logger.info(f"string chapter:{chapter_response.__str__(transcript=seg)}")
                    
                    if chapter_response is not None:
                        
                    
                        # Store the ChapterCreationResponse object for summary extraction
                        self.chapter_responses.append(chapter_response)
                        
                        # Store the transcript segment
                        self.chapter_transcripts.append(seg)
                        
                        # Create string representation with transcript only for indexing
                        chapter_response_str = chapter_response.__str__(transcript=seg)
                        self.chapter_response_strings.append(chapter_response_str)
                        
                        logger.info(f"Chapter {idx} response: {chapter_response_str}")
                        logger.info(f"Chapter {idx} created successfully.")
                    break
                except Exception as e:
                    attempts += 1
                    if attempts < max_attempts:
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        logger.info(
                            f"Chapter {idx} failed after {attempts} attempts."
                        )
            await asyncio.sleep(2)
        logger.info("Chapter Generation Completed!")
        
        
    async def _ingest(self, video_blob_url, youtube_url=None):
        url_info = f'BLOB={video_blob_url}\nYT_URL={youtube_url}'
        doc_objects: List[AISearchDocument] = [] 
        current_time = datetime.now()
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
                species=self.species_data['species'] or "None",
                variety=self.species_data['Variety_of_species'] or "None",
                blob_audio_url=self.blob_urls['audio_blob_url'].split(".net")[-1][1:] or "None",
                blob_video_url=video_blob_url.split(".net")[-1][1:] or "None",
                blob_transcript_file_url=self.blob_urls['transcript_blob_url'].split(".net")[-1][1:] or "None",
                blob_frames_folder_path=self.blob_urls['frames_blob_folder_url'].split(".net")[-1][1:] or "None",
                blob_timestamps_file_url=self.blob_urls['timestamps_blob_url'].split(".net")[-1][1:] or "None",
                blob_transcript_and_summary_file_url=self.blob_urls['transcript_and_summary_file_url'].split(".net")[-1][1:] or "None",
                embeddings=await self._create_embedding_normal(chapter_content_str)
            )
            doc_objects.append(obj)
            
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