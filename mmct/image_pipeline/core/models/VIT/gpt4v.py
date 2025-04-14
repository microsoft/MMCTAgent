import torch
from PIL import Image
import requests
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider, DefaultAzureCredential
import os
import io
import base64
import re
from dotenv import load_dotenv
load_dotenv(override=True)

class GPT4V:
    def __init__(self,api_key=None):
        if os.environ.get("TRAPI_RESOURCE_ENABLE")=='True':
          scope = "api://trapi/.default"
          credential = get_bearer_token_provider(AzureCliCredential(),scope)
          api_version = os.environ.get("TRAPI_API_VERSION")
          self.model = os.environ.get("TRAPI_AOPENAI_VISION_MODEL")  # Ensure this is a valid model name
          model_version = os.environ.get("TRAPI_AOPEN_VISION_MODEL_VERSION")  # Ensure this is a valid model version
          self.deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{self.model}_{model_version}')
          instance = os.environ.get('TRAPI_INSTANCE') # See https://aka.ms/trapi/models for the instance name
          endpoint = f"{os.environ.get('TRAPI_BASE_ENDPOINT')}/{instance}"
        else:
          scope = "https://cognitiveservices.azure.com/.default"
          credential = get_bearer_token_provider(DefaultAzureCredential(),scope)
          api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
          self.model = os.environ.get("AZURE_OPENAI_VISION_MODEL")
          model_version = os.environ.get("AZURE_OPENAI_VISION_MODEL_VERSION")
          self.deployment_name = self.model
          endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        self.client = client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
            )

    def convert_image(self, img_pil, save_path="output_image.jpg"):
        # Save the image locally
        #img_pil.save(save_path, format="JPEG")

        # Convert the image to base64
        image_bytes = io.BytesIO()
        img_pil.save(image_bytes, format="JPEG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        
        return base64_image
    
    def get_concat_h_resize(self, im1, im2):
        if im2.height > im1.height:
            ratio = im2.height/im1.height
            im1 = im1.resize((int(ratio*im1.width), int(ratio*im1.height)))
        else:
            ratio = im1.height/im2.height
            im2 = im2.resize((int(ratio*im2.width), int(ratio*im2.height)))
        
        padding = int(0.04*max(im1.width, im2.width))
        new_width = im1.width + padding + im2.width
        dst = Image.new('RGB', (new_width, max(im1.height, im2.height)))
        if im2.height>im1.height:
            pad_h = (im2.height-im1.height)//2
            dst.paste(im1, (0, pad_h))
            dst.paste(im2, (im1.width + padding, 0))
        else:
            pad_h = (im1.height-im2.height)//2
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width + padding, pad_h))
        return dst

    def run(self, prompt, images,selected_image=None, **kwargs):
        if isinstance(images, list):
            if selected_image.lower() == "left":
                im = images[0]
            elif selected_image.lower() == "right":
                im = images[1]
            else:
                im = self.get_concat_h_resize(*images)
        else:
            im = images
        response = self.client.chat.completions.create(
          
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
          model=self.deployment_name,
          temperature=0,
          max_tokens=500,
        )
        generated_text = response.choices[0].message.content
        return generated_text.strip()