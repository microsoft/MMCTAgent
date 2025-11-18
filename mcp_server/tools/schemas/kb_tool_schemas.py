from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class FilterParams(BaseModel):
    category: Optional[str] = Field(None, description="Video category filter, e.g. 'Motivational Speech'")
    sub_category: Optional[str] = Field(None, description="Video sub-category filter")
    subject: Optional[str] = Field(None, description="Subject filter")
    variety: Optional[str] = Field(None, description="Variety filter")
    time_from: Optional[str] = Field(
        None,
        description="ISO timestamp to filter videos from (inclusive), e.g. '2025-08-01T00:00:00Z'"
    )
    time_to: Optional[str] = Field(
        None,
        description="ISO timestamp to filter videos to (inclusive)"
    )
    hash_video_id: Optional[str] = Field(
        default=None,
        description="Hash Id for the particular video"
    )

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    query_type: str = Field(
        "full",
        description="Search mode: one of 'full', 'vector', 'semantic'"
    )
    index_name: str = Field(...,description="Azure Index Name")
    k: int = Field(10, description="Number of top results to return")
    filters: Optional[FilterParams] = None
    select: Optional[List[str]] = Field(
        default=None,
        description="List of field names to fetch from the index results. Available field names: [category, sub_category, subject, variety, hash_video_id ...]"
    )


async def get_filter_string(filters: Dict[str, Any]) -> Optional[str]:
    # clauses: List[str] = []
    filter_conditions = dict()

    category = filters.get('category')
    if category:
        cat_esc = category.replace("'", "''")
        filter_conditions['category'] = {"eq": cat_esc}

    sub = filters.get('sub_category')
    if sub and sub.lower() != 'none':
        sub_esc = sub.replace("'", "''")
        filter_conditions["sub_category"] = {'eq': sub_esc}

    subject = filters.get('subject')
    if subject:
        subj_esc = subject.replace("'", "''")
        filter_conditions["subject"] = {'eq': subj_esc}

    hash_video_id = filters.get('hash_video_id')
    if hash_video_id:
        hash_video_id_esc = hash_video_id.replace("'","''")
        filter_conditions["hash_video_id"] = {'eq': hash_video_id_esc}

    variety = filters.get('variety')
    if variety and variety.lower() != 'none':
        var_esc = variety.replace("'", "''")
        filter_conditions["variety"] = {'eq': var_esc}

    time_from = filters.get('time_from')
    if time_from:
        filter_conditions["time"] = {'ge': time_from}

    time_to = filters.get('time_to')
    if time_to:
        filter_conditions["time"] = {'le': time_to}

    if not filter_conditions:
        return None

    return filter_conditions
