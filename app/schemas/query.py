from pydantic import BaseModel, Field, field_validator
from mmct.image_pipeline import ImageQnaTools

class ImageQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, example="Describe the image")
    tools: list[str] = Field(
        ...,
        example=["object_detection", "vit", "ocr", "recog"],
    )
    use_critic_agent: bool = Field(..., example=True)
    stream: bool = Field(..., example=False)
    
    @field_validator("tools")
    def split_tools_str(cls, v):
        if isinstance(v, list) and len(v)==1:
            # Convert comma-separated string into list
            return [item.strip() for item in v[0].split(",") if item.strip()]
        if isinstance(v, list):
            return v
        raise ValueError("tools must be a list of strings or comma-separated string")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Simple usage",
                    "description": "Query text with ocr tool",
                    "value": {
                        "query": "What is written here?",
                        "tools": ["ocr"],
                        "use_critic_agent": False,
                        "stream": False
                    }
                },
                {
                    "summary": "Multiple tools usage",
                    "description": "Use object detection and recognition",
                    "value": {
                        "query": "Identify objects",
                        "tools": ["object_detection","recog"],
                        "use_critic_agent": True,
                        "stream": False
                    }
                }
            ]
        }
    }

class VideoQueryRequest(BaseModel):
    query: str
    index_name: str
    top_n: int = Field(..., ge=1)
    use_computer_vision_tool: bool
    use_critic_agent: bool
    stream: bool