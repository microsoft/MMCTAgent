"""Video ingestion pipeline architecture.

This module provides the core IngestionPipeline class which processes raw video 
files into structured knowledge within the MMCT system, including scene 
segmentation, transcription, and graph indexing.
"""

from typing import Optional, Annotated, Any
from loguru import logger

from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import (
    get_media_folder,
    get_file_hash,
)
from mmct.video_pipeline.core.ingestion.utils.helper import get_video_duration
from mmct.utils.logging_config import log_manager
from mmct.video_pipeline.core.ingestion.pipelines import (
    PipelineRunner,
    load_pipeline_config,
    get_default_ingestion_config,
    StepContext,
    StepDataStore,
)


class IngestionPipeline:
    """Orchestrates the ingestion of video files into the MMCT system.

    The IngestionPipeline handles the end-to-end process of preparing a video 
    for querying. It manages a sequence of processing steps (pipeline) that 
    extract metadata, segment the video, generate transcripts, and index 
    the content in a graph database.

    Attributes:
        video_path (str): Local path to the source video file.
        video_id (str): Unique hash-based identifier for the video.
        provider (Any): Bundle of provider implementations for various services.
        language (Languages): The primary language of the video content.
        url (str): Optional URL associated with the video.
        transcript_path (str): Optional path to a pre-existing transcript.
        pipeline_config_path (str): Optional path to a custom YAML pipeline config.
        frame_stacking_grid_size (int): Grid size for combining keyframes.
        save_local_report (bool): Whether to persist the execution report locally.
        verbosity (int): Logging level (0=Silent, 1=Info, 2=Debug).
        playlist_id (str): YouTube/Media playlist identifier.
        playlist_order (int): Position of the video within the playlist.
    """

    def __init__(
        self,
        video_path: Annotated[str, "Local path to the video file to be ingested"],
        video_id: Annotated[str, "Unique identifier (hash) for the video"],
        provider: Annotated[
            Any,
            "Provider bundle object containing all required providers as attributes",
        ],
        language: Annotated[
            Optional[Languages],
            "Language of the video (Languages Enum)",
        ] = None,
        url: Annotated[
            Optional[str], "Optional URL associated with the video for metadata enrichment"
        ] = None,
        transcript_path: Annotated[
            Optional[str],
            "Path to an existing transcript file (.srt); skips transcription if provided",
        ] = None,
        pipeline_config_path: Annotated[
            Optional[str],
            "Optional path to a custom pipeline configuration YAML file",
        ] = None,
        disable_console_log: Annotated[
            bool,
            "Boolean flag to disable console logs during ingestion (Deprecated, use verbosity=0)",
        ] = False,
        frame_stacking_grid_size: Annotated[
            int, "Grid size for frame horizontal stacking (>1 enables stacking, 1 disables)"
        ] = 4,
        save_local_report: Annotated[bool, "Whether to save the pipeline report locally"] = False,
        verbosity: Annotated[int, "Logging verbosity: 0=Progress Bar Only, 1=Info, 2=Debug"] = 0,
        playlist_id: Annotated[
            Optional[str],
            "Optional playlist ID this video belongs to (e.g. YouTube playlist ID)",
        ] = None,
        playlist_order: Annotated[
            Optional[int],
            "Optional 1-based position of this video within its playlist",
        ] = None,
    ):
        """Initializes the IngestionPipeline.

        Args:
            video_path: Local path to the video file.
            video_id: Unique identifier for the video, usually a file hash.
            provider: A bundle of provider instances (LLM, DB, Storage).
            language: The video's language (required if transcript_path is None).
            url: Source URL for metadata association.
            transcript_path: Path to an existing .srt file to skip transcription.
            pipeline_config_path: Path to a custom YAML pipeline configuration.
            disable_console_log: Legacy flag (deprecated in favor of verbosity).
            frame_stacking_grid_size: Grid size for tiled keyframes.
            save_local_report: If True, saves an execution report to disk.
            verbosity: Detail level of the logs (0, 1, or 2).
            playlist_id: Identifier for the parent playlist.
            playlist_order: Numeric order within the parent playlist.

        Raises:
            ValueError: If neither language nor transcript_path is provided.
            Exception: If configuration fetching fails.
        """
        try:
            # We delay config fetching logging until we set up the logger level
            pass
        except Exception as e:
            # Fallback logger if config fails immediately
            logger.exception(f"Exception occurred while fetching the MMCT config: {e}")
            raise Exception(f"Exception occurred while fetching the MMCT config: {e}")

        # Determine log level
        log_level = "WARNING"
        if verbosity == 1:
            log_level = "INFO"
        elif verbosity >= 2:
            log_level = "DEBUG"

        # Handle legacy disable_console_log if True, effectively silence or keep generic
        if disable_console_log:
            log_manager.disable_console()
        else:
            # Reset console to ensure level is correct
            log_manager.disable_console()
            log_manager.enable_console(level=log_level)

        self.logger = log_manager.get_logger()
        self.logger.info(
            "Successfully retrieved the MMCT config"
        )  # Will only show if verbosity >= 1

        # Validate that language is provided if transcript_path is not provided
        if not transcript_path and not language:
            raise ValueError("language parameter is required when transcript_path is not provided")

        self.video_path = video_path
        self.video_id = video_id
        self.provider = provider
        self.language = language
        self.url = url
        self.video_id = video_id
        self.transcript_path = transcript_path
        self.pipeline_config_path = pipeline_config_path
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.save_local_report = save_local_report
        self.original_video_path = video_path
        self.verbosity = verbosity
        self.playlist_id = playlist_id
        self.playlist_order = playlist_order

    async def run(self):
        """Executes the end-to-end ingestion pipeline.

        This method orchestrates the sequence of steps defined in the pipeline 
        configuration (load from config or default). It manages the context, 
        state, and reporting for the entire ingestion process.

        Returns:
            PipelineReport: A detailed report of the execution status and metrics.

        Raises:
            Exception: If the pipeline execution fails.
        """
        try:
            pipeline_config = None
            if self.pipeline_config_path:
                try:
                    pipeline_config = load_pipeline_config(self.pipeline_config_path)
                    self.logger.info(
                        f"Successfully loaded custom pipeline config from {self.pipeline_config_path}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load custom pipeline config from {self.pipeline_config_path}: {e}"
                    )
                    self.logger.info("Falling back to default ingestion config")

            if pipeline_config is None:
                pipeline_config = get_default_ingestion_config()

            # Calculate parent video metadata
            if not self.video_id:
                self.video_id = await get_file_hash(self.video_path)

            # Use original path for duration to be safe, or provided path
            video_duration = await get_video_duration(self.video_path)

            self.logger.info(f"Video ID: {self.video_id}, Duration: {video_duration:.2f}s")

            # Create StepContext for this execution
            self.context = StepContext(
                video_path=self.video_path,
                provider=self.provider,
                data_store=StepDataStore(),
                logger=self.logger,
                language=self.language,
                url=self.url,
                transcript_path=self.transcript_path,
                output_dir=await get_media_folder(),
                video_id=self.video_id,
                video_duration=video_duration,
                user_params={
                    "frame_stacking_grid_size": self.frame_stacking_grid_size,
                    "playlist_id": self.playlist_id,
                    "playlist_order": self.playlist_order,
                },
                save_local_report=self.save_local_report,
                verbosity=self.verbosity,
            )

            # Instantiate PipelineRunner
            runner = PipelineRunner(pipeline_config=pipeline_config, context=self.context)

            self.logger.info(f"Starting pipeline execution for {self.video_id}...")
            report = await runner.run()

            if report.status == "failed":
                self.logger.error("Pipeline failed.")
                raise Exception("Ingestion pipeline failed.")

            self.logger.info("Pipeline completed successfully!")
            return report

        except Exception as e:
            self.logger.exception(f"Exception occurred while running Ingestion pipeline: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    pass
