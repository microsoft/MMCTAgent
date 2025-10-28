"""
functionality of this tool is optical character recognition
"""
from mmct.image_pipeline.core.models.recog.mplug_base import MPLUGBase
from mmct.image_pipeline.core.models.recog.mplug_large import MPLUGLarge
from mmct.image_pipeline.core.models.recog.instructBlipCap import BlipCap
from PIL import Image
from typing_extensions import Annotated


async def recog_tool(img: Annotated[str, "path of image"],priority: Annotated[str,'There are 3 models of recognization tool which one to pick - 1 for small, 2 for Base,3 for Large']="3") -> str:
    """
    You can use this tool to analyze the given image, The tool should be used when
    you require to understand the scene in the image, and get a descriptive text
    about the image. The algorithm returns the description about the image in simple string.

    This returns response in string which is simply contains the description.
    input: 
        {}
    Input is always empty as it doesnt require anything as input and analyzes on the image that you are given. Always ignore the arguement priority and do not generate that in the input.

    response:
        The output is a string containing the description.
    """
    img = Image.open(img).convert("RGB")
    a = MPLUGBase() if priority == 1 else MPLUGLarge() if priority == 2 else BlipCap()
    resp = a(img)
    return resp