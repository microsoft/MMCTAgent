"""Dense export pipeline step.

Exports pipeline outputs to local files:
- Dense chapters, events, objects as JSON
- Interactive HTML graph visualization using pyvis

Output directory: {output_dir}/dense_export/{video_id}/
"""

from .step import DenseExportStep

__all__ = ["DenseExportStep"]
