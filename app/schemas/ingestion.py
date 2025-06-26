from pydantic import BaseModel
from mmct.video_pipeline import TranscriptionServices, Languages

class IngestionRequest(BaseModel):
    index_name: str
    transcription_service: TranscriptionServices
    language: Languages
    use_azure_cv_tool: bool