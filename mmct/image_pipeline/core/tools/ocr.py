"""Optical Character Recognition (OCR) tool.

This module provides a tool for extracting text from images using TrOCR models
of varying sizes (Small, Base, Large).
"""

from typing import Annotated
from PIL import Image
from mmct.image_pipeline.core.models.ocr.trocr_base import TROCRBase
from mmct.image_pipeline.core.models.ocr.trocr_small import TROCRSmall
from mmct.image_pipeline.core.models.ocr.trocr_large import TROCRLarge

class OcrTool:
    """Tool for performing Optical Character Recognition on images.

    This tool uses Transformer-based OCR models to extract handwritten or
    printed text from input images.

    Attributes:
        img_path (str): Local path to the image file to be analyzed.
    """

    def __init__(self, img_path: Annotated[str, "path of image"]):
        """Initializes the OcrTool.

        Args:
            img_path: Local path to the image to analyze.
        """
        self.img_path = img_path

    async def ocr_tool(
        self,
        priority: Annotated[
            str,
            "Select the OCR model: '1' for Small, '2' for Base, '3' for Large. Default is '3'.",
        ] = "3",
    ) -> str:
        """Extracts text from the image using the selected TrOCR model.

        Args:
            priority: Model selection priority string.
                "1" -> TrOCR-Small
                "2" -> TrOCR-Base
                "3" -> TrOCR-Large (Default)

        Returns:
            str: The text content extracted from the image.
        """
        img = Image.open(self.img_path).convert("RGB")
        model = (
            TROCRSmall()
            if priority == "1"
            else TROCRBase() if priority == "2" else TROCRLarge()
        )
        resp = await model(img)
        return resp
