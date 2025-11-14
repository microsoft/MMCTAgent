from mmct.providers.factory import provider_factory
from loguru import logger
import json
import uuid


class Cache:
    """
    Cache class for storing and retrieving question-answer pairs using semantic search.

    This class provides methods to:
    - Get cached responses based on semantic similarity of questions
    - Store new question-answer pairs with embeddings

    Attributes:
        search_provider: Provider for search operations
        embed_provider: Provider for generating embeddings
        index_name: Name of the search index to use (default: "gecko-cache-demo")
    """

    def __init__(self, index_name: str = "gecko-cache-demo"):
        """
        Initialize the Cache with search and embedding providers.

        Args:
            index_name: Name of the search index to use (default: "gecko-cache-demo")
        """
        self.index_name = index_name
        try:
            logger.info("Instantiating the embedding and search providers")
            self.search_provider = provider_factory.create_search_provider()
            self.embed_provider = provider_factory.create_embedding_provider()
            logger.info("Successfully instantiated the search and embedding providers")
        except Exception as e:
            logger.exception(f"Exception occurred while instantiating providers: {e}")
            raise

    async def get_cache_response(self, question: str) -> dict:
        """
        Retrieve a cached response for a given question using semantic search.

        Args:
            question: The query/question to search for in the cache

        Returns:
            dict: The most similar cached response containing answer, source, and videos
        """
        query_embedding = await self.embed_provider.embedding(question)
        fields_to_retrieve = ['answer', 'source', 'videos']

        search_results = await self.search_provider.search(
            query=question,
            index_name=self.index_name,
            search_text=None,
            query_type="semantic",
            top=1,
            select=fields_to_retrieve,
            embedding=query_embedding
        )

        return search_results[0]

    async def set_cache(self, question: str, answer: str, source: list, videos: list) -> bool:
        """
        Store a question-answer pair in the cache index.

        Args:
            question: The query/question to cache
            answer: The answer text (markdown formatted)
            source: List of source types (e.g., ["TEXTUAL", "VISUAL"])
            videos: List of video metadata dicts with hash_id, url, timestamps

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Caching response for question: {question}")

            # Generate embedding for the question
            question_embedding = await self.embed_provider.embedding(question)

            # Create document
            doc_id = str(uuid.uuid4())
            doc = {
                "id": doc_id,
                "question": question,
                "answer": answer,
                "source": source,
                "videos": json.dumps(videos),  # Store as JSON string
                "embeddings": question_embedding
            }

            # Upload to search index
            await self.search_provider.upload_documents(
                index_name=self.index_name,
                documents=[doc]
            )

            logger.info(f"Successfully cached response with ID: {doc_id}")
            return True

        except Exception as e:
            logger.exception(f"Error caching response: {e}")
            return False

if __name__ == "__main__":
    import asyncio

    async def test_cache():
        # Initialize Cache instance
        cache = Cache()

        # Test getting from cache
        test_question = "What are the required things for treating chili seeds and how do they benefit the growth process?"
        print(f"\n{'='*60}")
        print(f"Testing GET from cache")
        print('='*60)
        print(f"Query: {test_question}")
        result = await cache.get_cache_response(test_question)
        print(f"Results: {result}")

        # Test setting to cache
        print(f"\n{'='*60}")
        print(f"Testing SET to cache")
        print('='*60)
        new_question = "How to prepare organic fertilizer for vegetables?"
        new_answer = "To prepare organic fertilizer: mix vermicompost, cow urine, and natural ingredients."
        new_source = ["TEXTUAL", "VISUAL"]
        new_videos = [
            {
                "hash_id": "test123",
                "url": "https://www.youtube.com/watch?v=test",
                "timestamps": [["00:00:00", "00:01:00"]]
            }
        ]

        success = await cache.set_cache(new_question, new_answer, new_source, new_videos)
        print(f"Cache SET successful: {success}")

        # Properly close the providers to avoid aiohttp cleanup warnings
        if hasattr(cache.search_provider, 'close'):
            await cache.search_provider.close()
        if hasattr(cache.embed_provider, 'close'):
            await cache.embed_provider.close()

    asyncio.run(test_cache())
