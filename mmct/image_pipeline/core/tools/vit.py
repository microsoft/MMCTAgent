"""
This is a vit tool. it uses the vision_llm
"""

from mmct.image_pipeline.core.models.vit.visual_llm import VisualLLM
from PIL import Image
from typing_extensions import Annotated
from mmct.providers.base import BaseLLMProvider

class VitTool:
    def __init__(self, llm_provider: BaseLLMProvider, img_path: Annotated[str, "path of image"]):
        self.llm_provider = llm_provider
        self.img_path = img_path

    async def vit_tool(self, query: Annotated[str, "detailed/complete query about the image"]) -> str:
        """
        a advance visual tool which can describe image. it takes image path and query as input and the output is simple text answering the query given.
        """
        prompt = f"""You are an advanced Vision Language Model Tool specialized in image understanding, visual description and then solving the user queries based on the information provided in the text and image along with the world knowledge.

    Purpose:
   Your goal is to analyze and interpret visual information from images, combine with the textual information available from the query and provide precise, contextually relevant, and concise textual answers to user queries about those images.

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
        - Please provide the final answer and a clear, step-by-step explanation of the reasoning and assumptions that led to it
    >>>
    Query:
    {query}
    <<<
    """
        img = Image.open(self.img_path).convert("RGB")
        model = VisualLLM(llm_provider=self.llm_provider)
        resp = await model.run(images=img, prompt=prompt)
        return resp