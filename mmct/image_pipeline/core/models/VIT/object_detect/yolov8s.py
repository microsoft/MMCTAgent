from ultralytics import YOLO
import torch
from PIL import Image
import requests


class YOLOs:
    def __init__(self, device = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = device or "auto"
        self.model = YOLO('yolov8s.pt')

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
    
    def __call__(self, image):
        
        results = self.model(image)
        boxes = {}
        for result in results:
            bxcls = result.boxes.cls.detach().cpu().numpy()
            bxxywh = result.boxes.xywh.detach().cpu().numpy()
            for class_n, xywh in zip( bxcls, bxxywh  ):
                
                boxes[result.names[int(class_n.item())]] = xywh

        return boxes

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
    a = YOLOs()
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # prompt = "What is unusual about this image?"
    resp = a(image)
    print(resp)