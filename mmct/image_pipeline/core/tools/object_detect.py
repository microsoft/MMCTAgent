"""
functionality of this tool is object detection
"""

from mmct.image_pipeline.core.models.object_detect.yolov8s import YOLOs
import numpy as np
from typing_extensions import Annotated

class ObjectDetectTool:
    def __init__(self, img_path: Annotated[str, "local path of image"]):
        self.img_path = img_path   
        
    async def object_detect_tool(self) -> str:
        """
        Object Detection tool
        """
        model = YOLOs()
        resp = await model(self.img_path)

        # Ensure all numpy arrays in the response are converted to lists
        def serialize_response(response):
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
