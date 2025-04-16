from mmct.image_pipeline.core.models.recog.instructblip import BlipT5XXL
from PIL import Image
import requests

class BlipCap(BlipT5XXL):
    def __init__(self, device=None):
        super().__init__(device=device)

    def __call__(self, image):
        return super().__call__(image, prompt="Describe the image, and elements in it with breif detail within 100 words.")


if __name__ == "__main__":
    a = BlipCap()
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # prompt = "What is unusual about this image?"
    resp = a(image)
    print(resp)