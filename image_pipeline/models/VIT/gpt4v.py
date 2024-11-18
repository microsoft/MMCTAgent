from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from utils.content_moderation import safe_pipeline_execute, safe_step_execute
import torch
from PIL import Image
import requests
import openai
import os
import io
import base64

class GPT4V:
    def __init__(self, api_key=None, api_base=None, api_type=None, api_version=None, model=None, device = None):
        self.api_key = api_key or os.getenv("GPT4V_OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("GPT4V_OPENAI_API_BASE")
        self.api_type = api_type or os.getenv("GPT4V_OPENAI_API_TYPE")
        self.api_version = api_version or os.getenv("GPT4V_OPENAI_API_VERSION")
        self.model = model or os.getenv("GPT4V_OPENAI_API_MODEL")
        self.deployment = os.getenv("GPT4V_OPENAI_API_DEPLOYMENT")
        self.client = None
        if self.api_key is None and bool(os.getenv("GPT4V_OPENAI_MANAGED_IDENTITY", False)):
            from azure.identity import DefaultAzureCredential
            self.api_key = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default").token
            self.client = openai.AzureOpenAI(azure_endpoint=self.api_base, azure_deployment=self.deployment, azure_ad_token=self.api_key, api_version=self.api_version)
        elif self.api_type=="azure":
            self.client = openai.AzureOpenAI(azure_endpoint=self.api_base, azure_deployment=self.deployment, api_key=self.api_key, api_version=self.api_version)
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def convert_image(self, img_pil):
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
    
    @safe_step_execute(class_func=True)
    def __call__(self, prompt, images, selected_image=None, **kwargs):
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
          model=self.model,
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
          max_tokens=500,
        )
        generated_text = response.choices[0].message.content
        return generated_text.strip()

    def get_name(self):
        return """
                Vision Expert: vit\n
               """
    def get_desc(self):
        return """
                You can query information about the given image/images using simple natural language,
                This returns responses in simple language.
                input: 
                    {"query": "What is the number of objects in the image"}
                    or 
                    {"query": "What is the number of objects in the image", "selected_image": "1"}

                    The input can contain two values "query" and "selected_image". "selected_image" is optional but "query" is necessary for all queries.
                    "query" is to define the question that the Vision expert would answer about the image.
                    "selected_image" is used only when there are multiple images given in the problem setting. There are three valid options for "selected_image" i.e., "1", "2", "all". By default all is used, and for scenarios where there is only one image "selected_image" do not change the selection of image.

                response:
                    The output is simple text answering the query given.
               """
    def get_fn_schema(self):
        return """
               query: str
               selected_image: Optional[str] = "all" \n \t possible values: ["1","2",...(any number)...,"all"]
               """
    def __str__(self):
        return f"""
                {self.get_name()}
                {self.get_desc()}
                """