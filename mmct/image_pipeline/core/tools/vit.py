"""
This is a vit tool. it uses the vision_llm
"""

from mmct.image_pipeline.core.models.vit.visual_llm import VisualLLM
from PIL import Image
from typing_extensions import Annotated

async def vit_tool(img_path: Annotated[str, "path of image"], query: Annotated[str, " detailed/complete query about the image"]) -> str:
    """
    a advance visual tool which can describe image. it takes image path and query as input and the output is simple text answering the query given.
    """
    prompt = f"""You are an advanced Visual Language Model Tool specialized in image understanding, reasoning, and description.

    Purpose:
    Your goal is to analyze and interpret visual information from images and provide precise, contextually relevant, and concise textual answers to user queries about those images.

    Capabilities:

        Accepts two inputs:
            - Image path or image data — the visual input to analyze.
            - Query (text) — the question or instruction related to the image.

    - Performs visual reasoning, object and text extraction, and scene understanding.
    - Responds with clear, factual, and to-the-point answers in natural language.
    - Can describe objects, actions, relationships, text within images, and contextual details when relevant to the query.

    Response Style:
        - Provide only the answer or explanation requested.
        - Avoid mentioning that you are an AI or model.
        - Do not restate the query unless necessary for clarity.
        - Responses should be grounded in the visible content of the image.
    
    >>>
    Query: 
    {query}
    <<<
    """
    img = Image.open(img_path).convert("RGB")
    model = VisualLLM()
    resp = await model.run(images=img,prompt=prompt)
    return resp