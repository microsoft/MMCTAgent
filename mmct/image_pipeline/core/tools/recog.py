"""Image recognition and scene description tool.

This module provides a tool for analyzing scenes and generating descriptive
text for images using various recognition models (mPLUG, InstructBLIP).
"""

from typing import Annotated
from PIL import Image
from mmct.image_pipeline.core.models.recog.mplug_base import MPLUGBase
from mmct.image_pipeline.core.models.recog.mplug_large import MPLUGLarge
from mmct.image_pipeline.core.models.recog.instructBlipCap import BlipCap

class RecogTool:
    """Tool for recognizing objects and describing scenes in images.

    This tool uses advanced vision models to generate a comprehensive 
    description of an image's content, including objects, actions, and 
    environmental context.

    Attributes:
        img_path (str): Path to the image file to be analyzed.
    """

    def __init__(self, img_path: Annotated[str, "path of image"]):
        """Initializes the RecogTool.

        Args:
            img_path: Local path to the image to analyze.
        """
        self.img_path = img_path

    async def recog_tool(
        self,
        priority: Annotated[
            str,
            'Model selection priority: "1" for Base, "2" for Large, "3" for InstructBLIP'
        ] = "3"
    ) -> str:
        """Analyzes the image and returns a descriptive text summary.

        This method should be used when a general understanding of the scene
        or a detailed description of the visual content is required.

        Args:
            priority: Model selection priority string. 
                "1" -> mPLUG-Base
                "2" -> mPLUG-Large
                "3" -> InstructBLIP (Default)

        Returns:
            str: A natural language description of the image content.
        """
        img = Image.open(self.img_path).convert("RGB")
        model = MPLUGBase() if priority == "1" else MPLUGLarge() if priority == "2" else BlipCap()
        resp = model(img)
        return resp