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
        self.load_glossary()
        logger.info("Glossary Loaded")
        
    def load_glossary(self):
        self.glossary_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),r"Glossary.csv"))
        self.hindi_glossary = self.glossary_df['hindi_terms'].to_list()