"""Service layer for the ingestion endpoint."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
from loguru import logger

from mmct.utils.error_handler import ConfigurationException
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import IngestionPipeline
from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import get_file_hash


async def run_ingestion(
    video_path: Path,
    providers: Any,
    language_value: str,
    url: Optional[str] = None,
    transcript_path: Optional[str] = None,
    frame_stacking_grid_size: int = 4,
    save_local_report: bool = False,
    verbosity: int = 0,
    playlist_id: Optional[str] = None,
    playlist_order: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full MMCT ingestion pipeline for an uploaded video file.

    Resolves the Languages enum member from the BCP-47 string value supplied
    by the router, computes video_id via SHA-256 file hash, and delegates to
    IngestionPipeline.run().
    """
    try:
        # Resolve Languages enum member from BCP-47 string
        try:
            language = next(m for m in Languages if m.value == language_value)
        except StopIteration:
            raise HTTPException(status_code=400, detail=f"Unknown language code: {language_value!r}")

        video_id = await get_file_hash(str(video_path))

        pipeline = IngestionPipeline(
            video_path=str(video_path),
            video_id=video_id,
            provider=providers,
            language=language if not transcript_path else None,
            url=url,
            transcript_path=transcript_path,
            frame_stacking_grid_size=frame_stacking_grid_size,
            save_local_report=save_local_report,
            verbosity=verbosity,
            playlist_id=playlist_id,
            playlist_order=playlist_order,
        )

        report = await pipeline.run()

        return {
            "video_id": video_id,
            "status": report.status,
            "duration_seconds": getattr(report, "total_duration_seconds", None),
            "message": (
                "Ingestion pipeline completed successfully."
                if report.status == "completed"
                else "Ingestion pipeline encountered errors."
            ),
        }

    except HTTPException:
        raise
    except ConfigurationException as exc:
        raise HTTPException(status_code=500, detail=f"Configuration error: {exc}")
    except Exception as exc:
        logger.exception("Ingestion pipeline failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
