"""
Video Pipeline Module

This package exposes a minimal public API and uses lazy imports so that
heavy ingestion modules are imported only when actually needed.

- `VideoAgent` loads instantly.
- `IngestionPipeline`, `Languages`, and `TranscriptionServices` are loaded lazily.
"""

from typing import TYPE_CHECKING

__all__ = [
    "VideoAgent",
    "IngestionPipeline",
    "Languages",
    "TranscriptionServices",
]

# For type checkers and IDE autocompletion only (no runtime imports)
if TYPE_CHECKING:
    from .agents.video_agent import VideoAgent
    from .core.ingestion.ingestion_pipeline import IngestionPipeline
    from .core.ingestion.languages import Languages
    from .core.ingestion.transcription.transcription_services import (
        TranscriptionServices,
    )

# Lazy import map
_lazy_imports = {
    "VideoAgent": "agents.video_agent.VideoAgent",
    "IngestionPipeline": "core.ingestion.ingestion_pipeline.IngestionPipeline",
    "Languages": "core.ingestion.languages.Languages",
    "TranscriptionServices": "core.ingestion.transcription.transcription_services.TranscriptionServices",
}


def __getattr__(name):
    """Lazy import mechanism to avoid loading unused modules."""
    if name in _lazy_imports:
        module_path, attr = _lazy_imports[name].rsplit(".", 1)
        module = __import__(
            f".{module_path}", globals(), locals(), [attr], 1
        )
        value = getattr(module, attr)

        # Cache it so future accesses don’t re-import
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
