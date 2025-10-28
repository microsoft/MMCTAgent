"""
functionality of this tool is object detection
"""

from mmct.image_pipeline.core.models.object_detect.yolov8s import YOLOs
from PIL import Image
import numpy as np
from typing_extensions import Annotated


async def object_detect_tool(img: Annotated[str, "local path of image"]) -> str:
    """
    Object Detection tool
    """
    img = Image.open(img).convert("RGB")
    model = YOLOs()
    resp = await model(img)

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
