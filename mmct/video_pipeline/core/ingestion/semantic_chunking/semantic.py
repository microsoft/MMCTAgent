import warnings
import uuid
import os
import asyncio
from typing import List, Dict, Any, Union, Optional
from azure.search.documents import SearchClient
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
    def __init__(self, hash_id:str, index_name:str, transcript:str, base64Frames)->None:
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
        self.index_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), index_name=self.index_name, credential=self.token_provider)
        
    async def create_embedding_normal(self,text:str) -> List[float]:
        try:
            response = await self.embed_client.embeddings.create(
                input=[text], 
                model= os.getenv("AZURE_EMBEDDING_MODEL" if os.getenv("LLM_PROVIDER")=="azure" else "OPENAI_EMBEDDING_MODEL")
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f'Failed to create embedding:{e}',)

    async def pre_process(self):
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
        
        
    async def create_chapters(self):
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
        
        
    async def ingest(self, video_blob_url, youtube_url=None):
        url_info = f'BLOB={video_blob_url}\nYT_URL={youtube_url}'
        self.docs = []
        for txt in self.chapter_response_strings:  # Use strings with transcript for indexing
            self.docs.append({
                "id": str(uuid.uuid4()),
                "content": txt,
                "url": url_info,
                "species": self.species_data['species'],
                "variety": self.species_data['Variety_of_species'],
                "url_id": self.hash_id,
                "embeddings": await self.create_embedding_normal(txt)
            })
            
        self.index_client.upload_documents(documents=self.docs)
        
    async def run(self,video_blob_url, youtube_url=None):
        await self.pre_process()
        await self.create_chapters()
        await self.ingest(video_blob_url=video_blob_url, youtube_url=youtube_url)
        self.index_client.close()
        return self.chapter_responses, self.chapter_transcripts
        
if __name__=="__main__":
    semantic_chunker = SemanticChunking()
    asyncio.run(semantic_chunker.run())