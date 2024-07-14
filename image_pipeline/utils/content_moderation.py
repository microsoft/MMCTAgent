from azure.identity import DefaultAzureCredential
from azure.cognitiveservices.vision.contentmoderator import ContentModeratorClient
import datetime    
import msrest.authentication
from azure.identity import DefaultAzureCredential
from io import BytesIO
import base64
import enum
import json
import requests
from typing import Union, List, Dict
import os
from functools import wraps
from dotenv import load_dotenv
from PIL import Image
from types import SimpleNamespace

load_dotenv()


def sliding_window(text, max_chars=1024, overlap_words=2):
    words = text.split()
    windows = []
    start = 0
    while start < len(words):
        window = []
        current_length = 0
        for i in range(start, len(words)):
            word_length = len(words[i]) + 1 
            if current_length + word_length > max_chars:
                break
            window.append(words[i])
            current_length += word_length
        windows.append(' '.join(window))
        start += len(window) - overlap_words
    return windows

def merge_dicts(dict_a, dict_b):
    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        return [dict_a, dict_b] if dict_a != dict_b else dict_a

    merged = dict_a.copy()
    for key, value in dict_b.items():
        if key in dict_a:
            if isinstance(dict_a[key], dict) and isinstance(value, dict):
                merged[key] = merge_dicts(dict_a[key], value)
            elif isinstance(dict_a[key], list):
                merged[key] += value if isinstance(value, list) else [value]
            elif isinstance(value, list):
                merged[key] = [dict_a[key]] + value
            elif isinstance(dict_a[key], (int, float, str)) and isinstance(value, (int, float, str)):
                merged[key] = [dict_a[key], value] if dict_a[key] != value else dict_a[key]
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged

def optimal_size(size, min_limit=(50, 50), max_limit=(2048, 2048)):
    width, height = size
    min_width, min_height = min_limit
    max_width, max_height = max_limit

    if min_width <= width <= max_width and min_height <= height <= max_height:
        return size

    scale_to_min = max(min_width / width, min_height / height)
    scale_to_max = min(max_width / width, max_height / height)

    if width < min_width or height < min_height:
        new_width, new_height = width * scale_to_min, height * scale_to_min
    elif width > max_width or height > max_height:
        new_width, new_height = width * scale_to_max, height * scale_to_max
    else:
        return size
    new_width = max(min_width, min(new_width, max_width))
    new_height = max(min_height, min(new_height, max_height))

    return int(new_width), int(new_height)

    
def resize_base64_image(base64_string, size=None):
    imgdata = base64.b64decode(base64_string)
    img = Image.open(BytesIO(imgdata))
    if size is None:
        size = optimal_size(img.size)
    img = img.resize(size)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

class AzureCognitiveServicesCredential(msrest.authentication.Authentication):
    def __init__(self):
        self._credential = DefaultAzureCredential()
        self.token = None
        self.expiry = None

    def refresh_token(self):
        token_response = self._credential.get_token("https://cognitiveservices.azure.com/.default")
        self.token = token_response.token
        self.expiry = token_response.expires_on

    def signed_session(self, session=None):
        if not self.token or self.expiry <= datetime.datetime.now().timestamp():
            self.refresh_token()

        session = session or super(AzureCognitiveServicesCredential, self).signed_session()

        session.headers['Authorization'] = 'Bearer ' + self.token
        return session

class ContentModeratorDEF:
    def __init__(self, endpoint):
        auth = AzureCognitiveServicesCredential()
        self.client = ContentModeratorClient(endpoint, auth)
        
    def text_detect(self, text):
        results = None
        for windows in sliding_window(text, 1023, 0):
            if len(windows)<=110:
                windows += " " + "_"*(110-len(windows))
            text_fd = BytesIO(windows.encode('utf-8'))
            screen = self.client.text_moderation.screen_text(
                    text_content_type="text/plain",
                    text_content=text_fd,
                    language="eng",
                    autocorrect=True,
                    pii=True
                )
            if results is None:
                results = screen.as_dict()
            else:
                results = merge_dicts(results, screen.as_dict())
            
        return results

    def image_detect(self, img):
        img_stream = BytesIO()
        img.save(img_stream, format="PNG") 
        img_byte_arr = img_stream.getvalue()
        image_features = ["Adult", "Racy", "Gore", "Text"]
        evaluation = self.client.image_moderation.evaluate_file_input(
            image_stream=BytesIO(img_byte_arr),
            data_representation="application/octet-stream",
            mimetype="image/png",
            additional_image_features=image_features
            # custom_headers={"Content-Type": "application/octet-stream"}
        )
        # print(evaluation.as_dict())
        return evaluation

    def ocr_detect(self, img):
        img_stream = BytesIO()
        size = optimal_size(img.size)
        img = img.resize(size)
        img.save(img_stream, format="PNG") 
        img_byte_arr = img_stream.getvalue()
        img_byte_arr = img_byte_arr
        try:
            ocr = self.client.image_moderation.ocr_file_input(
                image_stream=BytesIO(img_byte_arr),
                language="eng",
                data_representation="application/octet-stream",
                mimetype="image/png",
            )
        except Exception as e:
            ocr = {"text": ""}
            ocr = SimpleNamespace(**ocr)
        return ocr


class MediaType(enum.Enum):
    Text = 1
    Image = 2


class Category(enum.Enum):
    Hate = 1
    SelfHarm = 2
    Sexual = 3
    Violence = 4


class Action(enum.Enum):
    Accept = 1
    Reject = 2


class DetectionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        """
        Exception raised when there is an error in detecting the content.

        Args:
        - code (str): The error code.
        - message (str): The error message.
        """
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"DetectionError(code={self.code}, message={self.message})"


class Decision(object):
    def __init__(
        self, suggested_action: Action, action_by_category: Dict[Category, Action]
    ) -> None:
        """
        Represents the decision made by the content moderation system.

        Args:
        - suggested_action (Action): The suggested action to take.
        - action_by_category (Dict[Category, Action]): The action to take for each category.
        """
        self.suggested_action = suggested_action
        self.action_by_category = action_by_category


class ContentSafety(object):
    def __init__(self, endpoint: str, subscription_key: str = None, api_version: str = None,  auth_token: str = None) -> None:
        """
        Creates a new ContentSafety instance.

        Args:
        - endpoint (str): The endpoint URL for the Content Safety API.
        - subscription_key (str): The subscription key for the Content Safety API.
        - api_version (str): The version of the Content Safety API to use.
        """
        self.endpoint = endpoint
        self.subscription_key = subscription_key
        self.api_version = api_version
        self.auth_token = auth_token

    def build_url(self, media_type: MediaType) -> str:
        """
        Builds the URL for the Content Safety API based on the media type.

        Args:
        - media_type (MediaType): The type of media to analyze.

        Returns:
        - str: The URL for the Content Safety API.
        """
        if media_type == MediaType.Text:
            return f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"
        elif media_type == MediaType.Image:
            return f"{self.endpoint}/contentsafety/image:analyze?api-version={self.api_version}"
        else:
            raise ValueError(f"Invalid Media Type {media_type}")

    def build_headers(self) -> Dict[str, str]:
        """
        Builds the headers for the Content Safety API request.

        Returns:
        - Dict[str, str]: The headers for the Content Safety API request.
        """
        if not self.subscription_key:
            return {
                "Authorization": "Bearer "+ self.auth_token,
                "Content-Type": "application/json",
            }
        else:
            return {
                "Ocp-Apim-Subscription-Key": self.subscription_key,
                "Content-Type": "application/json",
            }

    def build_request_body(
        self,
        media_type: MediaType,
        content: str,
        blocklists: List[str],
    ) -> dict:
        """
        Builds the request body for the Content Safety API request.

        Args:
        - media_type (MediaType): The type of media to analyze.
        - content (str): The content to analyze.
        - blocklists (List[str]): The blocklists to use for text analysis.

        Returns:
        - dict: The request body for the Content Safety API request.
        """
        if media_type == MediaType.Text:
            return {
                "text": content,
                "blocklistNames": blocklists,
            }
        elif media_type == MediaType.Image:
            return {"image": {"content": content}}
        else:
            raise ValueError(f"Invalid Media Type {media_type}")

    def detect(
        self,
        media_type: MediaType,
        content: str,
        blocklists: List[str] = [],
    ) -> dict:
        """
        Detects unsafe content using the Content Safety API.

        Args:
        - media_type (MediaType): The type of media to analyze.
        - content (str): The content to analyze.
        - blocklists (List[str]): The blocklists to use for text analysis.

        Returns:
        - dict: The response from the Content Safety API.
        """
        url = self.build_url(media_type)
        headers = self.build_headers()
        if media_type == MediaType.Image:
            content = resize_base64_image(content)
            request_body = self.build_request_body(media_type, content, blocklists)
            payload = json.dumps(request_body)
            
            response = requests.post(url, headers=headers, data=payload)

            res_content = response.json()

            if response.status_code != 200:
                raise DetectionError(
                    res_content["error"]["code"], res_content["error"]["message"]
                )

            return res_content
        elif media_type == MediaType.Text:
            results = None
            for windows in sliding_window(content, 9990, 0):
                if len(windows)<=110:
                    windows += " " + "_"*(110-len(windows))
                request_body = self.build_request_body(media_type, windows, blocklists)
                payload = json.dumps(request_body)

                response = requests.post(url, headers=headers, data=payload)
                res_content = response.json()
                if response.status_code != 200:
                    raise DetectionError(
                        res_content["error"]["code"], res_content["error"]["message"]
                    )
                
                if results is None:
                    results = res_content
                else:
                    results = merge_dicts(results, res_content)

            return results

    def get_detect_result_by_category(
        self, category: Category, detect_result: dict
    ) -> Union[int, None]:
        """
        Gets the detection result for the given category from the Content Safety API response.

        Args:
        - category (Category): The category to get the detection result for.
        - detect_result (dict): The Content Safety API response.

        Returns:
        - Union[int, None]: The detection result for the given category, or None if it is not found.
        """
        category_res = detect_result.get("categoriesAnalysis", None)
        for res in category_res:
            if category.name == res.get("category", None):
                return res
        raise ValueError(f"Invalid Category {category}")

    def make_decision(
        self,
        detection_result: dict,
        reject_thresholds: Dict[Category, int],
    ) -> Decision:
        """
        Makes a decision based on the Content Safety API response and the specified reject thresholds.
        Users can customize their decision-making method.

        Args:
        - detection_result (dict): The Content Safety API response.
        - reject_thresholds (Dict[Category, int]): The reject thresholds for each category.

        Returns:
        - Decision: The decision based on the Content Safety API response and the specified reject thresholds.
        """
        action_result = {}
        final_action = Action.Accept
        for category, threshold in reject_thresholds.items():
            if threshold not in (-1, 0, 2, 4, 6):
                raise ValueError("RejectThreshold can only be in (-1, 0, 2, 4, 6)")

            cate_detect_res = self.get_detect_result_by_category(
                category, detection_result
            )
            if cate_detect_res is None or "severity" not in cate_detect_res:
                raise ValueError(f"Can not find detection result for {category}")

            severity = cate_detect_res["severity"]
            action = (
                Action.Reject
                if threshold != -1 and severity >= threshold
                else Action.Accept
            )
            action_result[category] = action
            if action.value > final_action.value:
                final_action = action

        if (
            "blocklistsMatch" in detection_result
            and detection_result["blocklistsMatch"]
            and len(detection_result["blocklistsMatch"]) > 0
        ):
            final_action = Action.Reject

        # print(final_action.name)
        # print(action_result)

        return Decision(final_action, action_result)
   

    def shield_prompt(self,
            user_prompt: str,
            documents: list
        ) -> dict:
        """
        Detects unsafe content using the Content Safety API.

        Args:
        - user_prompt (str): The user prompt to analyze.
        - documents (list): The documents to analyze.

        Returns:
        - dict: The response from the Content Safety API.
        """
        api_version = "2024-02-15-preview"
        url = f"{self.endpoint}/contentsafety/text:shieldPrompt?api-version={api_version}"
        headers = self.build_headers()
        if len(user_prompt) <= 110:
            user_prompt += " " + "_"*(110-len(user_prompt))
        if isinstance(documents, list) and len(documents) > 0:
            for i in range(len(documents)):
                if len(documents[i]) <= 110:
                    documents[i] += " " + "_"*(110-len(documents[i]))
        else:
            if len(documents) <= 110:
                documents += " " + "_"*(110-len(documents))
        data = {
            "userPrompt": user_prompt,
            "documents": documents
        }
        response =  requests.post(url, headers=headers, json=data)
        return response.json()
    
    def protected_content_detection(self, text: str) -> dict:
        """
        Detects protected content using the Content Safety API.
        
        Args:
        - text (str): The text to analyze.
        
        Returns:
        - dict: The response from the Content Safety API.
        """
        api_version = "2023-10-15-preview"
        url = f"{self.endpoint}/contentsafety/text:detectProtectedMaterial?api-version={api_version}"
        headers = self.build_headers()
        results = None
        for windows in sliding_window(text, 990, 0):
            if len(windows)<=110:
                windows += " " + "_"*(110-len(windows))
            data = {
                "text": windows
            }
            response = requests.post(url, headers=headers, json=data)
            res_content = response.json()
            if response.status_code != 200:
                raise DetectionError(
                    res_content["error"]["code"], res_content["error"]["message"]
                )
            if results is None:
                results = res_content
            else:
                results = merge_dicts(results, res_content)
        return results
       

class ContentModMMCT:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super(ContentModMMCT, cls).__new__(cls)
        return cls._instance
        # if cls._instance is None:
        #     cls._instance = super(ContentModMMCT, cls).__new__(cls)
        # return cls._instance
    
    def __init__(self, content_moderation_endpoint, content_safety_endpoint, content_safety_multimodal_access = False):
        
        if content_moderation_endpoint is None or content_safety_endpoint is None:
            self.exclude_content_moderation = True
            return
        self.content_moderation = ContentModeratorDEF(content_moderation_endpoint)
        creds = DefaultAzureCredential()
        token = creds.get_token("https://cognitiveservices.azure.com/.default").token
        api_version = "2023-10-01"
        self.content_safety = ContentSafety(content_safety_endpoint,api_version=api_version, auth_token=token)
        self.text_threshold = {
            Category.Hate: 2,
            Category.SelfHarm: 4,
            Category.Sexual: 2,
            Category.Violence: 4,
        }
        self.image_threshold = {
            Category.Hate: 2,
            Category.SelfHarm: 4,
            Category.Sexual: 2,
            Category.Violence: 4,
        }
        self.content_safety_multimodal_access = content_safety_multimodal_access
        
        
    def convert_to_base64(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def check_attack_detected(self, result):
        if isinstance(result, dict):
            for value in result.values():
                if isinstance(value, bool) and value:
                    return True
                elif isinstance(value, dict):
                    if self.check_attack_detected(value):
                        return True
        elif isinstance(result, list):
            for item in result:
                if self.check_attack_detected(item):
                    return True
        return False
    
    def pipeline_check(self, query, img):
        if self.exclude_content_moderation:
            return False, "No Improper content detected"
        
        ocr = self.content_moderation.ocr_detect(img)
        
        ## JailBreak detection
        user_query = query
        document_text = ocr.text
        
        jailbreak_results = self.content_safety.shield_prompt(user_query, document_text)
        

        attack_detected = self.check_attack_detected(jailbreak_results)
        if attack_detected:
            return True, "JailBreak Detected"
        
        attack_detected = False
        query_protected_content_results = self.content_safety.protected_content_detection(user_query)
        attack_detected = attack_detected or self.check_attack_detected(query_protected_content_results)
        if len(document_text) > 0:
            document_protected_content_results = self.content_safety.protected_content_detection(document_text)
            attack_detected = attack_detected or self.check_attack_detected(document_protected_content_results)
        
        if attack_detected:
            return True, "Protected Content Detected in the input sample"
        
        pii_detected = False
        query_text_analysis_results = self.content_moderation.text_detect(user_query)
        pii_detected = len(query_text_analysis_results.get("pii",{})) > 0
        if len(document_text) > 0:
            document_text_analysis_results = self.content_moderation.text_detect(document_text)
            pii_detected = pii_detected or len(document_text_analysis_results.get("pii",{})) > 0
        
        if pii_detected:
            return True, "Personally Identifiable Information Detected in the input sample"
        
        ## Text Detection
        user_query_detection_result = self.content_safety.detect(MediaType.Text, user_query, [])
        user_query_decision = self.content_safety.make_decision(user_query_detection_result, self.text_threshold)
        if len(document_text) > 0:
            document_detection_result = self.content_safety.detect(MediaType.Text, document_text, [])
            document_decision = self.content_safety.make_decision(document_detection_result, self.text_threshold)
        else:
            document_decision = user_query_decision
            
        if user_query_decision.suggested_action == Action.Reject or document_decision.suggested_action == Action.Reject:
            return True, "Text Detected Improper content"
        
        ## Image Detection
        img_b64 = self.convert_to_base64(img)
        img_detection_result = self.content_safety.detect(MediaType.Image, img_b64, [])
        img_decision = self.content_safety.make_decision(img_detection_result, self.image_threshold)
        if img_decision.suggested_action == Action.Reject:
            return True, "Image Detected Improper content"
        
        return False, "No Improper content detected"

    def step_check(self, query, img):
        if self.exclude_content_moderation:
            return False, "No Improper content detected"
        
        tool_query_result = self.content_safety.detect(MediaType.Text, query, [])
        tool_query_decision = self.content_safety.make_decision(tool_query_result, self.text_threshold)
        
        if tool_query_decision.suggested_action == Action.Reject:
            return True, "Tool Query Detected Improper content"

        if self.content_safety_multimodal_access:
            raise Exception("Multimodal test run not implemented")
        
        attack_detected = False
        query_protected_content_results = self.content_safety.protected_content_detection(query)
        attack_detected = attack_detected or self.check_attack_detected(query_protected_content_results)
    
        if attack_detected:
            return True, "Protected Content Detected in intermediate query"
        
        pii_detected = False
        query_text_analysis_results = self.content_moderation.text_detect(query)
        pii_detected = len(query_text_analysis_results.get("pii",{})) > 0
        
        if pii_detected:
            return True, "Personally Identifiable Information Detected in intermediate query"
        
        return False, "Safe Input"
    
    def output_check(self, out):
        if self.exclude_content_moderation:
            return False, "No Improper content detected"
        
        output_result = self.content_safety.detect(MediaType.Text, out, [])
        output_decision = self.content_safety.make_decision(output_result, self.text_threshold)
        
        if output_decision.suggested_action == Action.Reject:
            return True, "Output Detected Improper content"
        
        attack_detected = False
        query_protected_content_results = self.content_safety.protected_content_detection(out)
        attack_detected = attack_detected or self.check_attack_detected(query_protected_content_results)
    
        if attack_detected:
            return True, "Protected Content Detected in Tool/LLM Output"
        
        pii_detected = False
        query_text_analysis_results = self.content_moderation.text_detect(out)
        pii_detected = len(query_text_analysis_results.get("pii",{})) > 0
        
        if pii_detected:
            return True, "Personally Identifiable Information Detected in intermediate query"
        
        return False, "Safe Output" 
    
    
def create_content_moderation_mmct_singleton_object():
    content_moderation_mmct = os.environ.get("CONTENT_MODERATION_MMCT", "False").lower() == "true"
    
    if not content_moderation_mmct:
        return ContentModMMCT(None, None)
    
    content_moderation_endpoint = os.environ.get("CONTENT_MODERATION_ENDPOINT")
    content_safety_endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    content_safety_multimodal_access = os.environ.get("CONTENT_SAFETY_MULTIMODAL_ACCESS", "False").lower() == "true"
    return ContentModMMCT(content_moderation_endpoint, content_safety_endpoint, content_safety_multimodal_access)

content_mod_mmct = create_content_moderation_mmct_singleton_object()    
        
def safe_pipeline_execute(class_func=False):
    global content_mod_mmct
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            offset = 1 if class_func else 0
            query = args[1 + offset]
            img = args[0 + offset]
            detected, reason = content_mod_mmct.pipeline_check(query, img)
            if not detected:
                return func(*args, **kwargs)
            else:
                return reason
        return wrapper
    return decorator

def safe_step_execute(class_func=False):
    global content_mod_mmct
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            offset = 1 if class_func else 0
            query = args[0 + offset]
            img = args[1 + offset]
            detected, reason = content_mod_mmct.step_check(query, img)
            if not detected:
                out = func(*args, **kwargs)
                out_detected, out_reason = content_mod_mmct.output_check(str(out))
                if not out_detected:
                    return out
                else:
                    return out_reason
            else:
                return reason
        return wrapper
    return decorator

def safe_output(class_func=False):
    global content_mod_mmct
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            detected, reason = content_mod_mmct.output_check(str(out))
            if not detected:
                return out
            else:
                return reason
        return wrapper
    return decorator