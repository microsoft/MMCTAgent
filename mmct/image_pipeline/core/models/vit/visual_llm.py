import torch
from PIL import Image
import requests
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
import io
import base64
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class VisualLLM:
    def __init__(self):
        # Initialize configuration
        self.config = MMCTConfig()

        # Initialize vision provider
        self.llm_provider = provider_factory.create_llm_provider()

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
            result = await self.llm_provider.chat_completion(
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
                temperature=0,
            )
            generated_text = result['content']
            return generated_text.strip()
        except Exception as e:
            raise Exception(f"Exception occured while performing GPT 4v call: {e}")
