"""
This is a VIT tool. it uses the GPT4V.
"""

from mmct.image_pipeline.core.models.VIT.gpt4v import GPT4V
from PIL import Image
from typing_extensions import Annotated

async def VITTool(img_path: Annotated[str, "path of image"], query: Annotated[str, " detailed/complete query about the image"]) -> str:
    #query:Annotated[str,"query by user"]
    """
    a advance visual tool which can describe image. it takes image path and query as input and the output is simple text answering the query given.
    """
    img = Image.open(img_path).convert("RGB")
    a = GPT4V()
    resp = a.run(images=img,prompt=query)
    return resp