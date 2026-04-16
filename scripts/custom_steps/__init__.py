"""Custom ingestion steps for the local video ingestion script.

Importing this module registers the application-specific steps with the
MMCT pipeline framework.  These steps are **not** part of the core MMCT
library — they demonstrate how client applications can extend the
ingestion pipeline with their own processing logic.

Usage:
    # In your script, import this module before running the pipeline:
    import custom_steps  # registers uniform_frames & transcript_upload

    # Then reference them in your pipeline YAML or PipelineConfig.
"""

from .uniform_frames import UniformFrameExtractionStep  # noqa: F401
from .transcript_upload import TranscriptUploadStep  # noqa: F401
