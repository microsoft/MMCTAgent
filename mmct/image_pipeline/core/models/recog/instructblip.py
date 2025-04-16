from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests


class BlipT5XXL:
    def __init__(self, device = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = device or "auto"
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl", device_map = self.device_map)
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")

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
    
    def __call__(self, images, prompt, selected_image=None):
        if isinstance(images, list):
            if selected_image.lower() == "left":
                image = images[0]
            elif selected_image.lower() == "right":
                image = images[1]
            else:
                image = self.get_concat_h_resize(*images)
        else:
            image = images
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text

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
                    {"query": "What is the number of objects in the image", "selected_image": "left"}

                    The input can contain two values "query" and "selected_image". "selected_image" is optional but "query" is neccessary for all queries.
                    "query" is to define th question that the Vision expert would answer about the image.
                    "selected_image" is used only when there are multiple images given in the problem setting. There are three valid options for "selected_image" i.e., "left", "right", "all". By default all is used, and for scenarios where there is only one image "selected_image" do not change the selection of image.

                response:
                    The output is simple text answering the query given.
               """
    def get_fn_schema(self):
        return """
                query: str
                selected_image: Optional[str] = "both" \n \t possible values: ["left","right","both"]
               """
    def __str__(self):
        return f"""
                {self.get_name()}
                {self.get_desc()}
               """



if __name__ == "__main__":
    a = BlipT5XXL()
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "What is unusual about this image?"
    resp = a(image,prompt)
    print(resp)