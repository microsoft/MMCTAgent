import pandas as pd
import os
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from mmct.video_pipeline.core.ingestion.languages import Languages
# Load environment variables
load_dotenv(find_dotenv(),override=True)

class Transcription:
    def __init__(self, video_path:str, hash_id:str, language:Languages = None):
        self.video_path = video_path
        self.hash_id = hash_id
        if language is None:
            self.source_language = {'lang':None,'lang-code':None}
        else:
            self.source_language = {'lang':language.name,'lang-code':language.value}
