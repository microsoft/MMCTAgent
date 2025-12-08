import os
from typing import Optional, Annotated, Dict
from loguru import logger

from mmct.config.providers import IngestionProviderConfig
from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import (
    get_file_hash,
    get_media_folder,
)
from mmct.video_pipeline.core.ingestion.utils.helper import (
    get_video_duration,
)
from mmct.utils.logging_config import log_manager
from mmct.video_pipeline.core.ingestion.pipelines import (
    PipelineRunner,
    load_pipeline_config,
    get_default_ingestion_config,
    StepContext,
    StepDataStore,
)


class IngestionPipeline:
    """
    IngestionPipeline handles the ingestion to prepare it for use with the VideoAgent system.
    Refactored to use the new step-based pipeline framework.
    """

    def __init__(
        self,
        video_path: Annotated[str, "Local path to the video file to be ingested"],
        provider: Annotated[
            IngestionProviderConfig,
            "Configuration object containing all service providers",
        ],
        language: Annotated[
            Optional[Languages],
            "Language of the video (Languages Enum), required only when transcript_path is not provided",
        ] = None,
        url: Annotated[
            Optional[str], "Optional URL associated with the video for metadata enrichment"
        ] = None,
        transcript_path: Annotated[
            Optional[str],
            "Path to an existing transcript file (.srt); skips transcription if provided",
        ] = None,
        disable_console_log: Annotated[
            bool, "Boolean flag to disable console logs during ingestion"
        ] = False,
        frame_stacking_grid_size: Annotated[
            int, "Grid size for frame horizontal stacking (>1 enables stacking, 1 disables)"
        ] = 4,
        save_local_report: Annotated[bool, "Whether to save the pipeline report locally"] = False,
    ):
        try:
            logger.info("Successfully retrieved the MMCT config")
        except Exception as e:
            logger.exception(f"Exception occurred while fetching the MMCT config: {e}")
            raise Exception(f"Exception occurred while fetching the MMCT config: {e}")

        if disable_console_log == False:
            log_manager.enable_console()
        else:
            log_manager.disable_console()
        self.logger = log_manager.get_logger()

        # Validate that language is provided if transcript_path is not provided
        if not transcript_path and not language:
            raise ValueError("language parameter is required when transcript_path is not provided")

        self.video_path = video_path
        self.provider = provider
        self.language = language
        self.url = url
        self.transcript_path = transcript_path
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.save_local_report = save_local_report
        self.original_video_path = video_path

    async def run(self):
        """Main ingestion pipeline method using the new PipelineRunner."""
        try:
            pipeline_config = get_default_ingestion_config()

            # Calculate parent video metadata
            video_id = await get_file_hash(self.video_path)
            # Use original path for duration to be safe, or provided path
            video_duration = await get_video_duration(self.video_path)

            self.logger.info(f"Video ID: {video_id}, Duration: {video_duration:.2f}s")

            # Create StepContext for this execution
            context = StepContext(
                video_path=self.video_path,
                provider=self.provider,
                data_store=StepDataStore(),
                logger=self.logger,
                language=self.language,
                url=self.url,
                transcript_path=self.transcript_path,
                output_dir=await get_media_folder(),
                video_id=video_id,
                video_duration=video_duration,
                user_params={
                    "frame_stacking_grid_size": self.frame_stacking_grid_size,
                },
                save_local_report=self.save_local_report,
            )

            # Instantiate PipelineRunner
            runner = PipelineRunner(pipeline_config=pipeline_config, context=context)

            self.logger.info(f"Starting pipeline execution for {video_id}...")
            report = await runner.run()

            if report.status == "failed":
                self.logger.error("Pipeline failed.")
                raise Exception("Ingestion pipeline failed.")

            self.logger.info("Pipeline completed successfully!")

        except Exception as e:
            self.logger.exception(f"Exception occurred while running Ingestion pipeline: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    pass
