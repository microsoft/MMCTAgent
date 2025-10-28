"""
This is a vit tool. it uses the vision_llm
"""

from mmct.image_pipeline.core.models.vit.gpt4v import GPT4V
from PIL import Image
from typing_extensions import Annotated

async def vit_tool(img_path: Annotated[str, "path of image"], query: Annotated[str, " detailed/complete query about the image"]) -> str:
    """
    a advance visual tool which can describe image. it takes image path and query as input and the output is simple text answering the query given.
    """
    img = Image.open(img_path).convert("RGB")
    a = GPT4V()
    resp = await a.run(images=img,prompt=query)
    return resp