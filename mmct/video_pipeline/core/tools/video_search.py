# Importing modules
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from azure.identity.aio import DefaultAzureCredential
from loguru import logger
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from mmct.video_pipeline.core.ingestion.models import SpeciesVarietyResponse
from typing_extensions import Annotated
from typing import Optional
import asyncio
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class VideoSearch:
    def __init__(self, query, index_name, top_n=3, min_threshold=80):
        self.query = query
        self.top_n = top_n
        self.index_name = index_name
        self.min_threshold = min_threshold
        
        # Initialize configuration
        self.config = MMCTConfig()
        
        # Initialize providers
        self.llm_provider = provider_factory.create_llm_provider(
            self.config.llm.provider,
            self.config.llm.model_dump()
        )
        
        # Try to create embedding provider, fallback to LLM config if embedding config is incomplete
        try:
            self.embedding_provider = provider_factory.create_embedding_provider(
                self.config.embedding.provider,
                self.config.embedding.model_dump()
            )
        except Exception as e:
            logger.warning(f"Failed to create embedding provider: {e}. Using LLM provider for embeddings.")
            # Use LLM config for embedding provider as fallback
            llm_config = self.config.llm.model_dump()
            # Add embedding-specific deployment name if available
            if hasattr(self.config.llm, 'embedding_deployment_name') and self.config.llm.embedding_deployment_name:
                llm_config['deployment_name'] = self.config.llm.embedding_deployment_name
            elif hasattr(self.config.llm, 'deployment_name') and self.config.llm.deployment_name:
                llm_config['deployment_name'] = self.config.llm.deployment_name
            
            self.embedding_provider = provider_factory.create_embedding_provider(
                self.config.llm.provider,
                llm_config
            )
        
        # Try to create search provider, but handle missing configuration gracefully
        try:
            self.search_provider = provider_factory.create_search_provider(
                self.config.search.provider,
                self.config.search.model_dump()
            )
        except Exception as e:
            logger.warning(f"Failed to create search provider: {e}. Search functionality may be limited.")
            self.search_provider = None

    def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            from azure.identity.aio import AzureCliCredential
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            asyncio.run(cli_credential.get_token("https://search.azure.com/.default"))
            return cli_credential
        except Exception:
            return DefaultAzureCredential()

    async def generate_embeddings(self, text: str):
        """Function to generate embeddings for the given text

        Args:
            text (str): input string

        Returns:
            [list]: OpenAI Embeddings
        """
        try:
            return await self.embedding_provider.embedding(text)
        except Exception as e:
            raise Exception(f"Exception occured while creating embeddings: {e}")

    async def Species_and_variety_query(self, transcript:str)->str:
        """
        Extract species and variety information from a video transcript using an AI model.

        Args:
            transcript (str): The text transcription of the video.

        Returns:
            str: A JSON-formatted string containing species and variety information, or error details.
        """
        try:
            system_prompt = f"""
            You are a TranscriptAnalyzer. Your job is to find all the details from the transcripts of every 2 seconds and from the audio.
            Mention only the English name or the text into the response. If the text mentioned in the video is in Hindi or any other language, then convert it into English.
            If any text from transcript is in Hindi or any other language, translate it into English and include it in the response.
            Topics to include in the response:
            1. Species name talked about in the video.
            2. Specific variety of species (e.g., IPA 15-06, IPL 203, IPH 15-03) discussed.
            If the transcript does not contain any species or variety, assign 'None'.
            Ensure the response language is only English, not Hinglish or Hindi or any other language.
            Include the English-translated name of species and their variety only if certain.
            """

            prompt = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The audio transcription is: {transcript}",
                        }
                    ],
                },
            ]

            response = await self.llm_provider.chat_completion(
                messages=prompt,
                temperature=0,
                response_format=SpeciesVarietyResponse,
            )
            # Get the parsed Pydantic model from the response
            parsed_response: SpeciesVarietyResponse = response["content"]
            # Return the model as JSON string
            return parsed_response.model_dump_json()
        except Exception as e:
            return SpeciesVarietyResponse(
            species="None", Variety_of_species="None"
            ).model_dump_json()

    async def search_ai(
        self,
        query,
        index_name,
        top_n,
        min_threshold,
        species=None,
        variety=None,
    ):
        try:
            min_threshold = min_threshold / 100
            
            # Generate embeddings for the query
            query_embds = await self.generate_embeddings(text=query)
            
            # Build filter expression
            filter_expression = []
            if species:
                filter_expression.append(f"species eq '{species}'")
            if variety:
                if variety != "None":
                    logger.info(f"setting filter for variety: {variety}")
                    filter_expression.append(f"variety eq '{variety}'")

            filter_query = " and ".join(filter_expression) if filter_expression else None
            
            # Use the search provider or fallback to direct Azure Search
            if self.search_provider:
                search_results = await self.search_provider.search(
                    query=query,
                    index_name=index_name,
                    search_text=None,
                    vector_queries=[VectorizedQuery(vector=query_embds, fields="embeddings")],
                    vector_filter_mode=VectorFilterMode.PRE_FILTER,
                    top=50,
                    filter=filter_query,
                    select=["species", "variety", "blob_video_url", "hash_video_id", "youtube_url"]
                )
            else:
                # Fallback to direct Azure Search implementation
                from azure.search.documents.aio import SearchClient
                from azure.core.credentials import AzureKeyCredential
                import os
                
                # Get search configuration from environment
                AZURE_MANAGED_IDENTITY = os.environ.get("MANAGED_IDENTITY", None)
                azure_search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT", None)
                
                if not azure_search_endpoint:
                    raise Exception("Azure search endpoint is missing in env!")
                
                if AZURE_MANAGED_IDENTITY is None:
                    raise Exception("MANAGED_IDENTITY requires boolean value")
                
                if AZURE_MANAGED_IDENTITY.upper() == "TRUE":
                    credential = self._get_credential()
                    index_client = SearchClient(
                        endpoint=azure_search_endpoint,
                        index_name=index_name,
                        credential=credential,
                    )
                else:
                    SEARCH_SERVICE_KEY = os.environ.get("SEARCH_SERVICE_KEY", None)
                    if SEARCH_SERVICE_KEY is None:
                        raise Exception("Azure Search Key is missing!")
                    index_client = SearchClient(
                        endpoint=azure_search_endpoint,
                        index_name=index_name,
                        credential=AzureKeyCredential(key=SEARCH_SERVICE_KEY),
                    )
                
                try:
                    vector_query = VectorizedQuery(vector=query_embds, fields="embeddings")
                    results = await index_client.search(
                        search_text=None,
                        vector_queries=[vector_query],
                        vector_filter_mode=VectorFilterMode.PRE_FILTER,
                        top=50,
                        filter=filter_query,
                        select=["species", "variety", "blob_video_url", "hash_video_id", "youtube_url"]
                    )
                    search_results = [dict(result) async for result in results]
                finally:
                    await index_client.close()
            
            # Filter results by minimum threshold
            filtered_results = []
            for result in search_results:
                if min_threshold <= result.get("@search.score", 0):
                    filtered_results.append(result)

            # Get top N unique results
            top_n_results = []
            seen_urls = set()
            for result in filtered_results:
                if result["blob_video_url"] not in seen_urls:
                    seen_urls.add(result["blob_video_url"])
                    top_n_results.append(result)
                if len(top_n_results) == top_n:
                    break
            return top_n_results
        except Exception as e:
            raise Exception(f"Error while doing AI search: {e}")

    async def query_search(self, query, index_name, top_n, min_threshold):
        try:
            response_url = []
            scores = []
            url_ids = []
            species_response = await self.Species_and_variety_query(query)
            species_response = eval(species_response)
            species = species_response.get("species", "None")
            variety = species_response.get("Variety_of_species", "None")
            if species == "None":
                species = None
            elif variety == "None":
                variety = None
                
            result = await self.search_ai(
                query,
                index_name,
                top_n=top_n,
                min_threshold=min_threshold,
                species=species,
                variety=variety,
            )
            for results in result:
                if results:
                    response_url.append({"BLOB":results["blob_video_url"], "YT_URL":results['youtube_url']})
                    scores.append(results["@search.score"])
                    url_ids.append(results["hash_video_id"])
            if not response_url:
                logger.info("Searching again")
                result = await self.search_ai(
                    query, index_name, top_n, min_threshold=min_threshold
                )
                for results in result:
                    response_url.append({"BLOB":results["blob_video_url"], "YT_URL":results['youtube_url']})
                    scores.append(results["@search.score"])
                    url_ids.append(results["hash_video_id"])
            return response_url, scores, url_ids
        except Exception as e:
            raise Exception(
                f"Exception occured while fetching top {top_n} results: {e}"
            )

    async def search(self):
        res = await self.query_search(
            query=self.query,
            index_name=self.index_name,
            top_n=self.top_n,
            min_threshold=self.min_threshold,
        )
        return {"video_id": res[-1], "video_url": res[0]}


async def video_search(
    query: Annotated[str, "query of which video id needs to fetch"],
    index_name: Annotated[str, "ai search index name"],
    top_n: Annotated[int, "n video_id retreivel"] = 1,
    llm_provider: Optional[object] = None,
    embedding_provider: Optional[object] = None,
    search_provider: Optional[object] = None
):
    """
    This tool returns the video id of ingested video corresponds to the query
    """
    video_search = VideoSearch(query=query, top_n=top_n, index_name=index_name, min_threshold=70)
    res = await video_search.search()
    return res


if __name__ == "__main__":
    # Example usage - replace with your actual values
    query = "example query"
    index_name = "your-index-name"
    top_n = 3
    res = asyncio.run(video_search(query=query, index_name=index_name, top_n=top_n))
    print(res)
