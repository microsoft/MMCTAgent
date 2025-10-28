import torch
from PIL import Image
import requests
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from mmct.llm_client import LLMClient  # Keep for backward compatibility
import os
import io
import base64
import re
import asyncio
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class GPT4V:
    def __init__(self, llm_provider=None, vision_provider=None):
        # Initialize configuration
        self.config = MMCTConfig()
        
        # Initialize providers
        if llm_provider is None:
            # Fall back to old pattern for backward compatibility
            service_provider = os.getenv("LLM_PROVIDER", "azure")
            self.client = LLMClient(service_provider=service_provider, isAsync=True).get_client()
            self.model_name = os.getenv(
                "LLM_MODEL_NAME"
                if os.getenv("LLM_PROVIDER") == "azure"
                else "OPENAI_MODEL_NAME"
            )
        else:
            # Use provider pattern
            self.llm_provider = llm_provider
            self.vision_provider = vision_provider or llm_provider
            self.model_name = self.config.llm.model_name
            self.client = LLMClient(service_provider=self.config.llm.provider, isAsync=True).get_client()

    def convert_image(self, img_pil):
        # Convert the image to base64
        image_bytes = io.BytesIO()
        img_pil.save(image_bytes, format="JPEG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        return base64_image

    def get_concat_h_resize(self, im1, im2):
        if im2.height > im1.height:
            ratio = im2.height / im1.height
            im1 = im1.resize((int(ratio * im1.width), int(ratio * im1.height)))
        else:
            ratio = im1.height / im2.height
            im2 = im2.resize((int(ratio * im2.width), int(ratio * im2.height)))

        padding = int(0.04 * max(im1.width, im2.width))
        new_width = im1.width + padding + im2.width
        dst = Image.new("RGB", (new_width, max(im1.height, im2.height)))
        if im2.height > im1.height:
            pad_h = (im2.height - im1.height) // 2
            dst.paste(im1, (0, pad_h))
            dst.paste(im2, (im1.width + padding, 0))
        else:
            pad_h = (im1.height - im2.height) // 2
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width + padding, pad_h))
        return dst

    async def run(self, prompt, images, selected_image=None, **kwargs):
        try:
            if isinstance(images, list):
                if selected_image.lower() == "left":
                    im = images[0]
                elif selected_image.lower() == "right":
                    im = images[1]
                else:
                    im = self.get_concat_h_resize(*images)
            else:
                im = images
            response = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""{prompt}""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.convert_image(im)}",
                                },
                            },
                        ],
                    }
                ],
                model=self.model_name,
                temperature=0,
            )
            generated_text = response.choices[0].message.content
            return generated_text.strip()
        except Exception as e:
            raise Exception(f"Exception occured while performing GPT 4v call: {e}")
