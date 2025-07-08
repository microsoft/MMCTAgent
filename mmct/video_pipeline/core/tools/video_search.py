# Importing modules
from openai import AzureOpenAI
from openai import ContentFilterFinishReasonError, RateLimitError, APITimeoutError
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from loguru import logger
from mmct.llm_client import LLMClient
from mmct.video_pipeline.core.ingestion.models import SpeciesVarietyResponse
from typing_extensions import Annotated
import os
import asyncio
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class VideoSearch:
    def __init__(self, query, index_name, top_n=3, min_threshold=80):
        self.query = query
        self.top_n = top_n
        self.index_name = index_name
        self.min_threshold = min_threshold
        service_provider = os.getenv("LLM_PROVIDER", "azure")
        self.openai_client = LLMClient(service_provider=service_provider, isAsync=True).get_client()

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
            return DefaultAzureCredential()

    async def generate_embeddings(self, text: str):
        """Function to generate embeddings for the given text

        Args:
            text (str): input string


        Returns:
            [list]: OpenAI Embeddings
        """
        try:
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=os.getenv(
                    "EMBEDDING_SERVICE_MODEL_NAME"
                    if os.getenv("LLM_PROVIDER") == "azure"
                    else "OPENAI_EMBEDDING_MODEL_NAME"
                ),
            )
            return response.data[0].embedding
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
            You are a TranscriptAnalyzerGPT. Your job is to find all the details from the transcripts of every 2 seconds and from the audio.
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

            response = await self.openai_client.beta.chat.completions.parse(
                model=os.getenv(
                    "LLM_MODEL_NAME"
                    if os.getenv("LLM_PROVIDER") == "azure"
                    else "OPENAI_MODEL_NAME"
                ),
                messages=prompt,
                temperature=0,
                response_format=SpeciesVarietyResponse,
            )
            # Get the parsed Pydantic model from the response
            parsed_response: SpeciesVarietyResponse = response.choices[0].message.parsed
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
            # setting up the environment variables and required azure clients
            AZURE_MANAGED_IDENTITY = os.environ.get(
                "MANAGED_IDENTITY", None
            )
            azure_search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT", None)
            if azure_search_endpoint is None:
                raise Exception("Azure search endpoint is missing in env!")

            if AZURE_MANAGED_IDENTITY is None:
                raise Exception(
                    "MANAGED_IDENTITY requires boolean value for selecting authorization either with Managed Identity or API Key"
                )

            if AZURE_MANAGED_IDENTITY.upper() == "TRUE":
                # Use Azure CLI credential if available, fallback to DefaultAzureCredential
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
            query_embds = await self.generate_embeddings(text=query)
            vector_query = VectorizedQuery(vector=query_embds, fields="embeddings")

            filter_expression = []

            if species:
                filter_expression.append(f"species eq '{species}'")
            if variety:
                if variety != "None":
                    logger.info(f"setting filter for variety: {variety}")
                    filter_expression.append(f"variety eq '{variety}'")

            if filter_expression:
                filter_query = " and ".join(filter_expression)
            else:
                filter_query = None
                
            results = await index_client.search(
                search_text=None,
                vector_queries=[vector_query],
                vector_filter_mode=VectorFilterMode.PRE_FILTER,
                top=50,
                filter=filter_query,
                select=["species", "variety", "blob_video_url", "hash_video_id", "youtube_url"]
            )
            

            filtered_results = []
            async for result in results:
                if min_threshold <= result["@search.score"]:
                    filtered_results.append(result)

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
        finally:
            if index_client:
                await index_client.close()

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
    search_provider=None
):
    """
    This tool returns the video id of ingested video corresponds to the query
    """
    video_search = VideoSearch(query=query, top_n=top_n, index_name=index_name, min_threshold=70)
    res = await video_search.search()
    return res


if __name__ == "__main__":
    query = "What is the recommended timeframe for the first weeding after sowing the Virat species of mung?"
    index_name = "jharkhand-video-index"
    top_n = 3
    res = asyncio.run(video_search(query=query, index_name=index_name, top_n=top_n))
    print(res)
