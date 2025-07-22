import asyncio
import os
import aiofiles
from mmct.video_pipeline.core.ingestion.transcription.base_transcription import (
    Transcription,
)
from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import extract_mp3_from_video, get_media_folder
from dotenv import load_dotenv, find_dotenv
from loguru import logger
# Load environment variables
load_dotenv(find_dotenv(), override=True)


class WhisperTranscription(Transcription):
    def __init__(self, video_path: str, hash_id: str) -> None:
        super().__init__(video_path=video_path, hash_id=hash_id)
        self.local_save = []

    async def load_audio(self):
        try:
            self.audio_path = os.path.join(
                await get_media_folder(), f"{self.hash_id}.mp3"
            )
            logger.info("Initialized the audio path to save the extracted audio")
            await extract_mp3_from_video(
                video_path=self.video_path, output_path=self.audio_path
            )
            logger.info("Extracted the audio from the video")
            self.local_save.append(self.audio_path)
            logger.info(f"Saved the audio to : {self.audio_path}")
        except Exception as e:
            logger.exception(f"Error loading audio, {e}")
            raise

    async def get_transcript_whisper(self) -> str:
        """Extracts audio from a video file and transcribes it using Azure OpenAI Whisper."""
        try:
            model = os.getenv(
                "SPEECH_SERVICE_DEPLOYMENT_NAME"
                if os.getenv("LLM_PROVIDER") == "azure"
                else "OPENAI_SPEECH_SERVICE_MODEL_NAME"
            )
            logger.info("Performing translation using openai whisper endpoint")
            with open(self.audio_path, "rb") as file:
                result = await self.openai_stt_client.audio.translations.create(
                    file=file, model=model, response_format="srt"
                )
            logger.info("Successfully retrieved the translated transcript using Whisper")
            base_path = self.audio_path.split(".mp3")[0]
            transcript_local_path = os.path.join(base_path, ".srt")
            self.local_save.append(transcript_local_path)
            return result
        except Exception as e:
            logger.exception(f"Error in speech-to-text conversion: {e}")
            raise
        
    async def __call__(self):
        await self.load_audio()
        transcript = await self.get_transcript_whisper()
        transcript_save_path = os.path.join(await get_media_folder(),f"transcript_{self.hash_id}.srt")
        async with aiofiles.open(
            transcript_save_path, "w", encoding="utf-8"
        ) as f:
            await f.write(transcript)
        return transcript, self.local_save


if __name__ == "__main__":
    # Example usage - replace with your actual values
    video_path = "path/to/your/video.mp4"
    hash_id = "example_hash_id"
    transcriber = WhisperTranscription(video_path=video_path, hash_id=hash_id)
    transcipt = asyncio.run(transcriber.run())
    print(transcipt)
