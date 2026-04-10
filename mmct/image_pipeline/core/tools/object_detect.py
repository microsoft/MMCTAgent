"""Object detection tool using YOLOv8.

This module provides a tool for detecting and localizing objects within an image
using the YOLOv8 model.
"""

from typing import Annotated, Any
import numpy as np
from mmct.image_pipeline.core.models.object_detect.yolov8s import YOLOs

class ObjectDetectTool:
    """Tool for detecting and localizing objects in images.

    This tool identifies discrete objects and their corresponding bounding 
    boxes using an optimized YOLOv8 architecture.

    Attributes:
        img_path (str): Local path to the image file to be analyzed.
    """

    def __init__(self, img_path: Annotated[str, "local path of image"]):
        """Initializes the ObjectDetectTool.

        Args:
            img_path: Local path to the image to analyze.
        """
        self.img_path = img_path   
        
    async def object_detect_tool(self) -> Any:
        """Detects objects in the image and returns their labels and coordinates.

        Returns:
            Any: A serializable object (list or dict) containing detection results,
                including object classes and bounding box coordinates.
        """
        model = YOLOs()
        resp = await model(self.img_path)

        # Ensure all numpy arrays in the response are converted to lists
        def serialize_response(response):
            """Recursively converts numpy arrays to lists for JSON serialization."""
            if isinstance(response, np.ndarray):
                return response.tolist()
            elif isinstance(response, dict):
                return {k: serialize_response(v) for k, v in response.items()}
            elif isinstance(response, list):
                return [serialize_response(v) for v in response]
            else:
                return response

        # Convert the response to a serializable format
        serialized_resp = serialize_response(resp)
        return serialized_resp
