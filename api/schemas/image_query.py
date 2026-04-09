"""Schemas for the image query endpoint."""

from enum import Enum

from pydantic import BaseModel, Field


class ImageToolName(str, Enum):
    """Available analysis tools for the image query pipeline.

    Select one or more tools to use when analysing the uploaded image.
    Repeating the 'tools' query parameter enables multi-tool selection in Swagger UI.
    """

    vit = "vit"
    recog = "recog"
    object_detection = "object_detection"
    ocr = "ocr"


class ImageQueryResponse(BaseModel):
    """Response returned by POST /image-query."""

    response: str = Field(..., description="Answer to the image query")
    input_tokens: int = Field(..., description="Total input (prompt) tokens consumed")
    output_tokens: int = Field(..., description="Total output (completion) tokens consumed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "The image shows a restaurant menu with three sections: Starters, Mains, and Desserts. Under 'House Special' there are five items including Peking Duck and Dim Sum.",
                    "input_tokens": 320,
                    "output_tokens": 64,
                }
            ]
        }
    }
