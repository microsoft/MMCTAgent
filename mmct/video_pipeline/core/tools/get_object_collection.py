from typing import Annotated, List, Dict, Any, Optional
import json
import os
import difflib
from mmct.providers.base import BaseObjectCollectionVectorDBProvider
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class GetObjectCollection:
    def __init__(self, vectordb_object_registry: BaseObjectCollectionVectorDBProvider):
        self.vectordb_object_registry = vectordb_object_registry

    async def get_object_collection(
        self,
        object_names: Annotated[
            List[str], "extensive list of possible object names related to the query to retrieve"
        ],
        video_id: Annotated[Optional[str], "unique identifier for the video"] = None,
        url: Annotated[Optional[str], "url of the video"] = None,
    ) -> List[Dict[str, Any]]:
        """
        Description:
            Retrieve specific object details from the object collection.

            This tool fetches the object collection for a video and filters it to return only objects matching any of the provided `object_names`.

            IMPORTANT: This tool REQUIRES a valid video_id or URL before calling.
            - If video_id/URL is not provided in the instruction, call get_video_summary first to obtain it.
            - Use this tool for: Specific object queries, finding details about particular objects.

        Input Parameters:
            - object_names (List[str]): REQUIRED - A exhaustive list of possible object names to look for.
            - video_id (str): REQUIRED - Unique identifier for the video (obtain from get_video_summary if not provided)
            - url (str): REQUIRED if video_id not available - URL of the video

        Output:
            List of dictionaries containing details of the matching objects.
        """

        try:
            # Build filter conditions
            filter_conditions = dict()
            if url:
                filter_conditions["url"] = {"eq": url}
            elif video_id:
                filter_conditions["video_id"] = {"eq": video_id}

            # Search for object collection matching the filter
            results = await self.vectordb_object_registry.search(
                query="*",
                search_text="*",
                filter=filter_conditions,
                top=1,
            )

            matching_objects = []
            seen_objects = set()

            for document, score in results:
                doc_dict = document.model_dump()

                # Parse the object_collection JSON string
                object_collection_str = doc_dict.get("object_collection", "[]")
                try:
                    object_collection = json.loads(object_collection_str)
                except json.JSONDecodeError:
                    print(f"Error decoding object_collection JSON for video_id={video_id}")
                    continue

                if object_names:
                    normalized_queries = [name.lower() for name in object_names]

                    for obj in object_collection:
                        obj_name = obj.get("name", "").lower()
                        is_match = False

                        for query in normalized_queries:
                            # 1. Exact/Substring Match
                            if query in obj_name:
                                is_match = True
                                break  # Found a match for this object

                            # 2. Fuzzy Match if not direct match
                            similarity = difflib.SequenceMatcher(None, query, obj_name).ratio()
                            if similarity > 0.6:  # Threshold can be tuned
                                is_match = True
                                break  # Found a match for this object

                        if is_match:
                            # Use name as unique identifier to avoid duplicates
                            if obj.get("name") not in seen_objects:
                                matching_objects.append(obj)
                                seen_objects.add(obj.get("name"))
                else:
                    # If no object_names provided, return nothing to avoid bloat
                    pass

            return matching_objects

        except Exception as e:
            print(f"Error fetching object collection for video_id={video_id} or url={url}: {e}")
            return []
        finally:
            pass


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        video_id = "<hash-video-id>"
        # query = "<sample-query>" # Not used in V2
        object_name = "Christmas tree"

        # Mock provider for basic check if you were to run this file directly
        pass

    asyncio.run(main())
