from typing import Annotated, List, Dict, Any, Optional
import os
from mmct.providers.base import BaseObjectCollectionVectorDBProvider
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class GetObjectCollection:
    def __init__(self, vectordb_object_registry:BaseObjectCollectionVectorDBProvider):
        self.vectordb_object_registry = vectordb_object_registry
        
    async def get_object_collection(
        self,
        video_id: Annotated[Optional[str], "unique identifier for the video"] = None,
        url: Annotated[Optional[str], "url of the video"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Description:
            Retrieve object collection data including object descriptions, counts, and first_seen timestamps.

            This tool is used for object tracking and object-related queries.

            IMPORTANT: This tool REQUIRES a valid video_id or URL before calling.
            - If video_id/URL is not provided in the instruction, call get_video_summary first to obtain it.
            - Use this tool for: object counting, object tracking, object appearance details, first_seen timestamps.

        Input Parameters:
            - index_name (str): Name of the search index
            - video_id (str): REQUIRED - Unique identifier for the video (obtain from get_video_summary if not provided)
            - url (str): REQUIRED if video_id not available - URL of the video

        Output:
            List of dictionaries containing requested fields
        """
        
        try:
            # Build filter conditions
            filter_conditions = dict()
            if url:
                filter_conditions['url'] = {'eq': url}
            elif video_id:
                filter_conditions['video_id'] = {'eq': video_id}

            # Search for object collection matching the filter
            results = await self.vectordb_object_registry.search(
                query = "*",
                search_text = "*",
                filter = filter_conditions,
                top = 1,
            )
            fields_to_retrieve = ['object_collection','object_count','video_id']
            # Convert new return type List[Tuple[ObjectCollectionDocument, float]] to list of dicts
            results_with_scores = []
            for document, score in results:
                doc_dict = document.model_dump()

                # Only include fields specified by the user
                filtered_dict = {
                    field: doc_dict.get(field) for field in fields_to_retrieve
                    if field in doc_dict
                }
                filtered_dict['@search.score'] = score
                results_with_scores.append(filtered_dict)

            return results_with_scores

        except Exception as e:
            print(f"Error fetching object collection for video_id={video_id} or url={url}: {e}")
            return []
        finally:
            pass


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        index_name = "<index-name>"
        video_id = "<hash-video-id>"
        query = "<sample-query>"
        get_object_collection_object = GetObjectCollection()
        print(f"Fetching object collection for video_id: {video_id}")
        collection = await get_object_collection_object.get_object_collection(
            query=query,
            index_name=index_name,
            video_id=video_id
        )
        print(collection)

    asyncio.run(main())