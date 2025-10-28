"""
functionality of this tool is optical character recognition
"""

from mmct.image_pipeline.core.models.ocr.trocr_base import TROCRBase
from mmct.image_pipeline.core.models.ocr.trocr_small import TROCRSmall
from mmct.image_pipeline.core.models.ocr.trocr_large import TROCRLarge
from PIL import Image
from typing_extensions import Annotated


async def ocr_tool(
        img: Annotated[str, "Path to the input image file."],
        priority: Annotated[
            str,
            "Select the OCR model to use: '1' for Small, '2' for Base, '3' for Large. Default is '3'.",
        ],
    ) -> Annotated[str,"OCR results"]:
    """
    OCR Tool

    This function performs Optical Character Recognition (OCR) on the given image using a selected model size.
    """
    img = Image.open(img).convert("RGB")
    model = (
        TROCRSmall()
        if priority == "1"
        else TROCRBase() if priority == "2" else TROCRLarge()
    )
    resp = await model(img)
    return resp
