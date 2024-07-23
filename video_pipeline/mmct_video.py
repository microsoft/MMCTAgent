import os
import io
import re
import uuid
import argparse
import requests
import subprocess
import json
import time
import cv2
import pysrt
import base64
from PIL import Image
from tqdm import tqdm
from openai import AzureOpenAI
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

class VideoAgent:

    
    def extract_frames(self, fps=1):
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        frame_count = 0

        frames = []
        timestamps = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert the frame to PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame_pil)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            frame_count += 1

        cap.release()
        return frames, timestamps
    
    def moderate_video(self):
        # Azure endpoint
        moderation_endpoint = os.environ.get("AZURE_MODERATION_ENDPOINT")
        moderation_api_version = "2023-10-01"
        moderation_api_url = f"{moderation_endpoint}/contentsafety/image:analyze?api-version={moderation_api_version}"

        def moderate_image(image):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            payload = json.dumps({"image": {"content": img_str}})
            response = requests.post(moderation_api_url, headers=self.moderation_headers, data=payload)
            return response.json()

        moderated = False
        thresholds = {'Hate': 2, 'SelfHarm': 4, 'Sexual': 2, 'Violence': 4}

        # Select frames at 0.2 fps (every 5th frame if original is at 1 fps)
        frames_to_moderate = self.frames[::5]

        for frame in frames_to_moderate:
            result = moderate_image(frame)
            
            for category in result['categoriesAnalysis']:
                if category['severity'] >= thresholds[category['category']]:
                    moderated = True
                    break
            
            if moderated:
                break

        return moderated
    
    def moderate_text(self, text):
        moderation_endpoint = os.environ.get("AZURE_MODERATION_ENDPOINT")
        moderation_api_version = "2023-10-01"
        moderation_api_url = f"{moderation_endpoint}/contentsafety/text:analyze?api-version={moderation_api_version}"
        payload = json.dumps({"text": text})
        response = requests.post(moderation_api_url, headers=self.moderation_headers, data=payload)
        moderated = False
        thresholds = {'Hate': 2, 'SelfHarm': 2, 'Sexual': 4, 'Violence': 2}
        result = response.json()
        print(result)
        for category in result['categoriesAnalysis']:
            print(category['category'])
            if category['severity'] >= thresholds[category['category']]:
                moderated = True
                break
        return moderated
        
    
    def upload_video_to_blob(self):
        if self.blob_managed_identity:
            account_url = os.environ.get("BLOB_ACCOUNT_URL")
            # Initialize the BlobServiceClient
            blob_service_client = BlobServiceClient(account_url, credential=self.credential)
        else:
            # Get the connection string from an environment variable
            connect_str = os.environ.get("BLOB_CONNECTION_STRING")
            # Initialize the BlobServiceClient
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # Specify the container name
        container_name = os.environ.get("BLOB_CONTAINER_NAME")
        # Extract the video name from the file path
        video_name_in_blob = f"{self.session_id}.mp4"
        # Get the blob client for the specific blob
        self.blob_client = blob_service_client.get_blob_client(container=container_name, blob=video_name_in_blob)

        try:
            # Open the video file and upload it to the blob
            with open(self.video_path, "rb") as data:
                self.blob_client.upload_blob(data)
            blob_url = self.blob_client.url
            return blob_url
        except ResourceExistsError:
            # If the blob already exists, just log a message
            blob_url = self.blob_client.url
            return blob_url
        except Exception as e:
            print("Error uploading video to blob:", e)
            return None
    
    def create_index(self):
        index_url = f"{self.azurecv_endpoint}/computervision/retrieval/indexes/{self.session_id}?api-version=2023-05-01-preview"
        payload = {
            "features": [
                {
                    "name": "vision",
                    "domain": "generic"
                }
            ]
        }
        response = requests.put(index_url, headers=self.azurecv_headers, data=json.dumps(payload))
        return response.json()

    
    def add_video_to_index(self):
        ingestion_url = f"{self.azurecv_endpoint}/computervision/retrieval/indexes/{self.session_id}/ingestions/my-ingestion?api-version=2023-05-01-preview"
        if self.blob_managed_identity:
            document_url = self.blob_url
            payload = {
                "moderation": False,
                "videos": [
                    {
                    'mode': 'add',
                    'documentUrl': document_url
                    }
                ],
                "documentAuthenticationKind" : "managedIdentity"
            }
        else:
            sas_token = os.environ.get("BLOB_SAS_TOKEN")
            document_url = self.blob_url + "?" + sas_token
            payload = {
                "moderation": False,
                "videos": [
                    {
                    'mode': 'add',
                    'documentUrl': document_url
                    }
                ]
            }

        response = requests.put(ingestion_url, headers=self.azurecv_headers, data=json.dumps(payload))
        print("Video Addition Initialized:", response.json())

        check_url = f"{self.azurecv_endpoint}/computervision/retrieval/indexes/{self.session_id}/ingestions?api-version=2023-05-01-preview&$top=1"
        video_indexed = False
        while True:
            response = requests.get(check_url, headers=self.azurecv_headers)
            print("Video Addition Status:", response.json())
            if response.json()['value'][0]['state'] == "Completed":
                video_indexed = True
                print("Video Addition Completed")
                break
            elif response.json()['value'][0]['state'] == "Running":
                pass
            else:
                print("Error adding video to index")
                break
            time.sleep(10)
        return video_indexed
    
    def process_srt(self, batch_size=16):

        def get_embeddings(texts, model):
            processed_texts = [text.replace("\n", " ") for text in texts]
            response = self.client_embed.embeddings.create(input=processed_texts, model=model)
            return [data.embedding for data in response.data]
        
        subs = pysrt.open(self.transcript_path)
        processed_subs = []
        
        for i in tqdm(range(0, len(subs), batch_size)):
            batch = subs[i:i+batch_size]
            texts = [sub.text.replace('\n', ' ') for sub in batch]
            embeddings = get_embeddings(texts, model=os.getenv("ADA_EMBEDDING_DEPLOYMENT"))

            for j, sub in enumerate(batch):
                sub_dict = {
                    'start': sub.start.to_time().strftime("%H:%M:%S"),
                    'end': sub.end.to_time().strftime("%H:%M:%S"),
                    'text': texts[j],
                    'embedding': embeddings[j]
                }
                processed_subs.append(sub_dict)

        return processed_subs

    
    def __init__(self, video_path, transcript_path, system_prompt_path):

        self.video_path = video_path
        self.transcript_path = transcript_path
        self.system_prompt_path = system_prompt_path

        self.blob_managed_identity = os.environ.get("BLOB_MANAGED_IDENTITY") == "True"
        self.azurecv_managed_identity = os.environ.get("AZURECV_MANAGED_IDENTITY") == "True"
        self.azure_openai_managed_identity = os.environ.get("AZURE_OPENAI_MANAGED_IDENTITY") == "True"
        self.azure_moderation_managed_identity = os.environ.get("AZURE_MODERATION_MANAGED_IDENTITY") == "True"

        self.session_id = str(uuid.uuid4())

        if self.blob_managed_identity or self.azurecv_managed_identity or self.azure_openai_managed_identity or self.azure_moderation_managed_identity:
            self.credential = DefaultAzureCredential()
            self.token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
        video_base_name_with_extension = os.path.basename(self.video_path)
        self.video_base_name, _ = os.path.splitext(video_base_name_with_extension)

        print("Extracting frames...")
        self.frames, self.frame_timestamps = self.extract_frames()
        print("Done")
        print("")

        print("Uploading video to Azure Blob and creating index...")
        self.blob_url = self.upload_video_to_blob()
        print("Blob URL:", self.blob_url)

        self.azurecv_endpoint = os.environ.get("AZURECV_ENDPOINT")
        if self.azurecv_managed_identity:
            self.azurecv_headers = {
                "Authorization": "Bearer "+self.token.token,
                "Content-Type": "application/json"
            }
        else:
            self.azurecv_key = os.environ.get("AZURECV_KEY")
            self.azurecv_headers = {
                "Ocp-Apim-Subscription-Key": self.azurecv_key,
                "Content-Type": "application/json"
            }
        index_creation_response = self.create_index()
        print(index_creation_response)
        if 'error' not in index_creation_response:
            self.video_index = self.add_video_to_index()
        if self.video_index:
            print("Done")
        else:
            print("Video indexing failed")

        if self.azure_openai_managed_identity:
            self.client_embed = AzureOpenAI(
                                azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"), 
                                api_version="2023-03-15-preview",
                                azure_ad_token_provider=get_bearer_token_provider(self.credential, "https://cognitiveservices.azure.com/.default")
                                )
            self.client = AzureOpenAI(
                                azure_endpoint = os.getenv("AZURE_OPENAI_GPT4_ENDPOINT"), 
                                api_version="2023-03-15-preview",
                                azure_ad_token_provider=get_bearer_token_provider(self.credential, "https://cognitiveservices.azure.com/.default")
                                )
        else:
            self.client_embed = AzureOpenAI(
                                azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"), 
                                api_version="2023-03-15-preview",
                                api_key=os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
                                )
            self.client = AzureOpenAI(
                                azure_endpoint = os.getenv("AZURE_OPENAI_GPT4_ENDPOINT"), 
                                api_version="2023-03-15-preview",
                                api_key=os.getenv("AZURE_OPENAI_GPT4_KEY")
                                )
            
        self.gpt4v_api_base = os.getenv("AZURE_OPENAI_GPT4V_ENDPOINT")
        self.gpt4v_deployment_name = os.getenv("GPT4V_DEPLOYMENT")
        self.gpt4v_base_url = f"{self.gpt4v_api_base}/openai/deployments/{self.gpt4v_deployment_name}" 
        if self.azure_openai_managed_identity:
            self.gpt4v_headers = {   
                "Content-Type": "application/json",   
                "Authorization": "Bearer "+self.token.token 
            }
        else:
            self.gpt4v_api_key = os.environ.get("AZURE_OPENAI_GPT4V_KEY")
            self.gpt4v_headers = {   
                "Content-Type": "application/json",   
                "api-key": self.gpt4v_api_key 
            }

        self.gpt4v_endpoint = f"{self.gpt4v_base_url}/chat/completions?api-version=2023-12-01-preview"

        print("Processing transcript and generating embeddings...")
        self.transcript_embeddings = self.process_srt()
        print("Done")
        print("")

        if self.azure_moderation_managed_identity:
            self.moderation_headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token.token
            }
        else:
            self.moderation_key = os.environ.get("AZURE_MODERATION_KEY")
            self.moderation_headers = {
                "Ocp-Apim-Subscription-Key": self.moderation_key,
                "Content-Type": "application/json"
            }

        # Reading the system prompt
        with open(self.system_prompt_path, "r") as file:
            system_prompt = file.read()

        # Initializing messages list with system prompt
        self.messages = [{"role": "system", "content": system_prompt}]

        self.logs_reference = ""
        

    
    def get_transcript(self):
        with open(self.transcript_path, 'r') as file:
            return file.read()

    
    def query_transcript(self, transcript_query):

        def average_time(time_str1, time_str2):
            # Convert time strings to timedelta objects
            t1 = timedelta(hours=int(time_str1.split(":")[0]), minutes=int(time_str1.split(":")[1]), seconds=int(time_str1.split(":")[2]))
            t2 = timedelta(hours=int(time_str2.split(":")[0]), minutes=int(time_str2.split(":")[1]), seconds=int(time_str2.split(":")[2]))

            # Calculate the average
            avg_time = (t1 + t2) / 2

            # Convert the average timedelta back to time format
            total_seconds = int(avg_time.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
        
        query_embedding = self.client_embed.embeddings.create(input = [transcript_query], model=os.getenv("ADA_EMBEDDING_DEPLOYMENT")).data[0].embedding
        similarities = []
        for embedding in self.transcript_embeddings:
            similarities.append(cosine(query_embedding, embedding['embedding']))
        sorted_similarities = sorted(similarities)
        top_3_similarities = sorted_similarities[:3]
        top_3_indices = [similarities.index(similarity) for similarity in top_3_similarities]

        # Calculate the average of start and end times for the top 3 segments
        top_3_averaged_times = []
        for index in top_3_indices:
            start_time = self.transcript_embeddings[index]['start']
            end_time = self.transcript_embeddings[index]['end']
            averaged_time = average_time(start_time, end_time)
            top_3_averaged_times.append(averaged_time)

        # Convert the list to a comma-separated string
        return ", ".join(top_3_averaged_times)

    
    def query_frames_Azure_Computer_Vision(self, frames_query):
        search_url = f"{self.azurecv_endpoint}/computervision/retrieval/indexes/{self.session_id}:queryByText?api-version=2023-05-01-preview"
        search_payload = {
            "queryText": frames_query,
            "moderation": False,
            "top": 3
        }
        search_response = requests.post(search_url, headers=self.azurecv_headers, json=search_payload)
        response_data = search_response.json()

        results = []
        best_times = []
        for item in response_data.get("value", [])[:3]:
            best_time = item["best"]
            try:
                # Splitting the time and microsecond parts
                if '.' in best_time:
                    time_part, microsecond_part = best_time.split('.')
                    # Truncating or padding the microsecond part to six digits
                    microsecond_part = microsecond_part[:6].ljust(6, '0')
                    # Reconstructing the best_time string with the adjusted microsecond part
                    adjusted_best_time = f"{time_part}.{microsecond_part}"
                else:
                    adjusted_best_time = best_time
                # First, try parsing with microseconds
                formatted_best_time = datetime.strptime(adjusted_best_time, "%H:%M:%S.%f").strftime("%H:%M:%S")
            except ValueError:
                # If there's a ValueError, it means microseconds weren't present, so parse without them
                formatted_best_time = datetime.strptime(best_time, "%H:%M:%S").strftime("%H:%M:%S")
            best_times.append(formatted_best_time)
            results.append({
                "best": item["best"],
                "start": item["start"],
                "end": item["end"],
                "relevance": item["relevance"]
            })

        return ", ".join(best_times)
    
    
    def query_GPT4_Vision(self, timestamp, query):
        if ',' in timestamp:
            # If so, split the string at the comma and take the first part
            formatted_time_string = timestamp.split(',')[0]
        else:
            # If there's no comma, the format is already correct
            formatted_time_string = timestamp
        # Convert timestamp to milliseconds
        h, m, s = map(int, formatted_time_string.split(':'))
        timestamp_ms = (h * 3600 + m * 60 + s) * 1000

        # Find the nearest frame index
        nearest_frame_index = min(range(len(self.frame_timestamps)), key=lambda i: abs(self.frame_timestamps[i] - timestamp_ms))

        # Select frames
        start_index = max(0, nearest_frame_index - 4)
        end_index = min(nearest_frame_index + 5, len(self.frames) - 1)
        selected_frames = self.frames[start_index:end_index + 1]

        # # Encode selected frames to base64
        base64Frames = []
        for frame in selected_frames:
            buffered = io.BytesIO()
            frame.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64Frames.append(img_str)
            
        PROMPT_MESSAGES = [
            { "role": "system", "content": "You are `gpt-4-vision-preview`, the latest OpenAI model that can describe still frames provided by the user from a short video clip in extreme detail. The user has attached 10 frames sampled at 1 fps for you to analyze. You will never reply saying that you cannot see the images because the images are absolutely and always attached to the message. For every user query, you must carefully examine the frames for the relevant information corresponding to the user query and respond accordingly." }, 
            {
                "role": "user",
                "content": [
                    query,
                    *map(lambda x: {"image": x}, base64Frames),
                ],
            },
        ]
    
        data = { 
            "messages": PROMPT_MESSAGES, 
            "max_tokens": 500
        }
        try:
            response = requests.post(self.gpt4v_endpoint, headers=self.gpt4v_headers, data=json.dumps(data), timeout=180)
        except Exception as e:
            return "Error"

        response_json = response.json()
        if response.status_code == 200:
            return response_json["choices"][0]["message"]["content"]
        else:
            return "Error"
    
    
    def process_query(self, question):

        def replace_control_char_json_string(json_str):
            pattern = r'(?:"([^"]*)")'
            def replace_control_char(match):
                return '"' + match.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"'
            return re.sub(pattern, replace_control_char, json_str)

        def response_parser(gpt4_output):
            
            # Parse GPT-4 output assuming it's JSON formatted
            parsed_output = json.loads(replace_control_char_json_string(gpt4_output))

            if 'Answer' in parsed_output:
                # Final answer received
                return {'final_answer': True, 'answer': parsed_output['Answer'], 'assistant_response': gpt4_output}
            else:
                # Continue the loop with the next action
                tool_name = parsed_output['Action']['tool_name']
                tool_input = parsed_output['Action']['tool_input']

                # Call the appropriate tool based on tool_name and tool_input
                tool_output = getattr(self, tool_name)(**tool_input)
                return {'final_answer': False, 'output': json.dumps({'Output': tool_output}), 'assistant_response': gpt4_output}
            

        user_input = question
        print("User Input")
        print(user_input)

        # Append user input to messages list
        self.messages.append({"role": "user", "content": user_input})
        self.logs_reference+="User Input:\n"
        self.logs_reference+=user_input
        self.logs_reference+="\n"

        # Start reasoning loop with GPT-4
        while True:
            # Call GPT-4 API
            response = self.client.chat.completions.create(
                model=os.getenv("GPT4_32K_DEPLOYMENT"),
                messages=self.messages,
                temperature=0.0
            )

            # Get GPT-4 response and parse it
            gpt4_output = response.choices[0].message.content
            print("GPT-4 Output")
            print(gpt4_output)
            self.messages.append({"role": "assistant", "content": gpt4_output})
            self.logs_reference+="GPT-4 Output:\n"
            self.logs_reference+=gpt4_output
            self.logs_reference+="\n"

            parsed_response = response_parser(gpt4_output)
            if parsed_response['final_answer']:
                return parsed_response['answer'], self.logs_reference
            else:
                # Process the tool output and provide it to GPT-4
                tool_output = parsed_response['output']
                print("Tool Output")
                print(json.loads(replace_control_char_json_string(tool_output)))
                self.messages.append({"role": "user", "content": tool_output})
                self.logs_reference+="Tool Output:\n"
                self.logs_reference+=tool_output
                self.logs_reference+="\n"

    def critic(self):

        def stack_images_horizontally(frames):
            # Calculate the total width and the maximum height
            total_width = sum(image.width for image in frames)
            max_height = max(image.height for image in frames)

            # Create a new image with the appropriate size
            new_img = Image.new('RGB', (total_width, max_height))

            # Paste each frame into the new image
            x_offset = 0
            for img in frames:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width

            return new_img
        
        def replace_control_char_json_string(json_str):
            pattern = r'(?:"([^"]*)")'
            def replace_control_char(match):
                return '"' + match.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"'
            return re.sub(pattern, replace_control_char, json_str)

        pattern = r'"timestamp": "([0-9]{2}:[0-9]{2}:[0-9]{2})"'
        # Find all occurrences of the pattern in order
        timestamps1 = re.findall(pattern, self.logs_reference)

        pattern = r'"timestamp":"([0-9]{2}:[0-9]{2}:[0-9]{2})"'
        timestamps2 = re.findall(pattern, self.logs_reference)

        # New pattern to include timestamps with milliseconds
        pattern3 = r'"timestamp":"([0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3})"'
        timestamps3_full = re.findall(pattern3, self.logs_reference)

        # Removing the milliseconds part
        timestamps3 = [timestamp.split(',')[0] for timestamp in timestamps3_full]

        timestamps = timestamps1 + timestamps2 + timestamps3
        #print(timestamps)
        timestamps = timestamps[-10:] if len(timestamps) > 10 else timestamps

        if len(timestamps)>0:

            # Calculate frames to distribute and frames per timestamp
            total_frames = 10
            frames_per_timestamp = total_frames // len(timestamps)
            remainder_frames = total_frames % len(timestamps)

            assignment_str = ""  # Initialize the string for frame assignments
            base64Frames = []  # List to store encoded frames
            assignment_idx = 0
            for index, timestamp in enumerate(timestamps):
                h, m, s = map(int, timestamp.split(':'))
                timestamp_ms = (h * 3600 + m * 60 + s) * 1000

                # Find the nearest frame index
                nearest_frame_index = min(range(len(self.frame_timestamps)), key=lambda i: abs(self.frame_timestamps[i] - timestamp_ms))

                # Calculate the number of frames for this timestamp
                num_frames = frames_per_timestamp + (remainder_frames if index == len(timestamps) - 1 else 0)

                # Select frames
                start_index = max(0, nearest_frame_index - 5)
                end_index = min(nearest_frame_index + 4, len(self.frames) - 1)
                possible_frames = self.frames[start_index:end_index + 1]

                if len(possible_frames) < num_frames:
                    stack_size = 1
                    remainder = 0
                    selected_frames = []
                    for i in range(len(possible_frames)):
                        selected_frames.append(possible_frames[i])
                else:
                    stack_size = len(possible_frames) // num_frames
                    remainder = len(possible_frames) % num_frames
                    selected_frames = []

                    for i in range(num_frames):
                        frame_stack = possible_frames[i*stack_size:(((i + 1)*stack_size) + (remainder if i == num_frames - 1 else 0))]
                        stacked_frame = stack_images_horizontally(frame_stack)
                        selected_frames.append(stacked_frame)

                for idx, frame in enumerate(selected_frames):
                    buffered = io.BytesIO()
                    frame.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    base64Frames.append(img_str)


                # Update the assignment string
                if len(possible_frames) < num_frames:
                    assigned_frame_numbers = [str(i + 1) for i in range(assignment_idx, assignment_idx + len(possible_frames))]
                    assignment_idx += len(possible_frames)
                    assignment_str += f"Image(s) {', '.join(assigned_frame_numbers)} are for timestamp {timestamp}; "
                else:
                    assigned_frame_numbers = [str(i + 1) for i in range(assignment_idx, assignment_idx + num_frames)]
                    assignment_idx += num_frames
                    assignment_str += f"Image(s) {', '.join(assigned_frame_numbers)} are for timestamp {timestamp}; "

            assignment_str = assignment_str[:-2]  # Remove the trailing semicolon and space
            assignment_str += ". Note that each image may contain multiple horizontally stacked frames for that timestamp."

        # Reading the system prompt
        with open('./system_prompt_critic.txt', "r") as file:
            critic_system_prompt = file.read()

        # Initializing messages list with system prompt
        self.critic_messages = [{"role": "system", "content": critic_system_prompt}]
        
        # Add assignment string to logs message
        logs_message = json.dumps({'logs': self.logs_reference})

        if len(timestamps)>0:
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        logs_message,
                        *map(lambda x: {"image": x}, base64Frames),
                        assignment_str,
                    ],
                },
            ]
        else:
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        logs_message,
                    ],
                },
            ]

        data = { 
            "messages": self.critic_messages + PROMPT_MESSAGES, 
            "max_tokens": 4096
        }

        try:
            response = requests.post(self.gpt4v_endpoint, headers=self.gpt4v_headers, data=json.dumps(data), timeout=180)
        except Exception as e:
            print("Error in Critic")
            print(e)
            return "Error", "YES"

        if response.status_code == 200:
            response_json = response.json()
            response_content = response_json["choices"][0]["message"]["content"]
            print("\n\n\n")
            print("Critic Output:")
            print(response_content)
        else:
            print(response.json())
            return "Error", "YES"

        if "```json" in response_content:
            response_content = response_content.replace("```json\n", "").replace("\n```", "")
        parsed_response = json.loads(replace_control_char_json_string(response_content))
        if 'Feedback' and 'Verdict' in parsed_response:
            return json.dumps({'Critic Feedback': parsed_response['Feedback']}), parsed_response['Verdict']
        else:
            return "Error", "YES"
        
    def post_critic(self):

        def replace_control_char_json_string(json_str):
            pattern = r'(?:"([^"]*)")'
            def replace_control_char(match):
                return '"' + match.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"'
            return re.sub(pattern, replace_control_char, json_str)

        def response_parser(gpt4_output):
            
            # Parse GPT-4 output assuming it's JSON formatted
            parsed_output = json.loads(replace_control_char_json_string(gpt4_output))

            if 'Answer' in parsed_output:
                # Final answer received
                return {'final_answer': True, 'answer': parsed_output['Answer'], 'assistant_response': gpt4_output}
            else:
                # Continue the loop with the next action
                tool_name = parsed_output['Action']['tool_name']
                tool_input = parsed_output['Action']['tool_input']

                # Call the appropriate tool based on tool_name and tool_input
                tool_output = getattr(self, tool_name)(**tool_input)
                return {'final_answer': False, 'output': json.dumps({'Output': tool_output}), 'assistant_response': gpt4_output}

        critic_feedback, verdict = self.critic()
        self.logs_reference+="Critic Feedback:\n"
        self.logs_reference+=critic_feedback
        self.logs_reference+="\n"
        if verdict=="YES":
            return "Done", self.logs_reference
        self.messages.append({"role": "user", "content": critic_feedback})

        # Start reasoning loop with GPT-4
        while True:
            # Call GPT-4 API
            response = self.client.chat.completions.create(
                model=os.getenv("GPT4_32K_DEPLOYMENT"),
                messages=self.messages,
                temperature=0.0
            )

            # Get GPT-4 response and parse it
            gpt4_output = response.choices[0].message.content
            print("GPT-4 Output")
            print(json.loads(replace_control_char_json_string(gpt4_output)))
            self.messages.append({"role": "assistant", "content": gpt4_output})
            self.logs_reference+="GPT-4 Output:\n"
            self.logs_reference+=gpt4_output
            self.logs_reference+="\n"

            parsed_response = response_parser(gpt4_output)
            if parsed_response['final_answer']:
                return parsed_response['answer'], self.logs_reference
            else:
                # Process the tool output and provide it to GPT-4
                tool_output = parsed_response['output']
                print("Tool Output")
                print(json.loads(replace_control_char_json_string(tool_output)))
                self.messages.append({"role": "user", "content": tool_output})
                self.logs_reference+="Tool Output:\n"
                self.logs_reference+=tool_output
                self.logs_reference+="\n"

def save_transcript(video_path):
    video_base_name, _ = os.path.splitext(os.path.basename(video_path))
    video_base_path = os.path.dirname(video_path)
    # Function to extract audio from a local video file
    def extract_audio_from_local(video_path, output_path):
        try:
            with open(os.devnull, 'wb') as devnull:  # Suppress FFmpeg output
                subprocess.call(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', output_path], stdout=devnull, stderr=devnull)
        except Exception as e:
            print(f"An error occurred while extracting audio from {video_path}: {e}")

    extract_audio_from_local(video_path, f'{video_base_path}/{video_base_name}.mp3')
    if os.getenv("AZURE_OPENAI_MANAGED_IDENTITY")=="True":
        credential = DefaultAzureCredential()
        client = AzureOpenAI(
            azure_ad_token_provider=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),  
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        )
    else:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_WHISPER_KEY"),
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        )

    deployment_id = os.getenv("WHISPER_DEPLOYMENT") #This will correspond to the custom name you chose for your deployment when you deployed a model."

    result = client.audio.translations.create(
        file=open(f'{video_base_path}/{video_base_name}.mp3', "rb"),            
        model=deployment_id,
        response_format='srt'
    )
    with open(f'{video_base_path}/{video_base_name}.srt', 'w') as f:
        f.write(result)

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Video Question Answering System.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("question", type=str, help="Question to ask about the video")
    parser.add_argument("--critic", action="store_true", default=False, help="Whether to use critic or not (default: False)")
    parser.add_argument("--max_num_critic", type=int, default=1, help="Maximum number of critic calls (default: 1)")
    parser.add_argument("--moderation", type=bool, default=True, help="Whether to use content moderation (default: True)")

    # Parse the arguments
    args = parser.parse_args()

    video_base_name, _ = os.path.splitext(os.path.basename(args.video_path))
    video_base_path = os.path.dirname(args.video_path)
    save_transcript(args.video_path)
    transcript_path = f'{video_base_path}/{video_base_name}.srt'

    if args.moderation:
        system_prompt_path = "./system_prompt_planner_with_guardrails.txt"
    else:
        system_prompt_path = "./system_prompt_planner.txt"

    # Initialize the video agent with the provided paths
    agent = VideoAgent(args.video_path, transcript_path)
    
    if args.moderation:
        video_moderation = agent.moderate_video()
        print("Video Moderation:", video_moderation)
        question_moderation = agent.moderate_text(args.question)
        print("Question Moderation:", question_moderation)
        if video_moderation or question_moderation:
            result = "Blocked"
            logs = "Video or question contains inappropriate content"
            print(result)
            print("\n\nLog Details:\n")
            print(logs)
            exit()

    # Process the question on the video
    try:
        result, logs = agent.process_query(args.question)
        if args.moderation:
            answer_moderation = agent.moderate_text(result)
            print("Answer Moderation:", answer_moderation)
            if answer_moderation:
                result = "Blocked"
                logs = "Answer contains inappropriate content"
        print(result)
        print("\n\nLog Details:\n")
        print(logs)
    except Exception as e:
        result = "Error"
        logs = f"An error occurred: {e}"
        print(result)
        print("\n\nLog Details:\n")
        print(logs)

    check_whether_can_continue = result != "Blocked" and result != "Error"
    
    # Optionally run the critic method
    if args.critic and check_whether_can_continue:
        num_critic = 0
        check = ""
        while check != "Done" and num_critic < args.max_num_critic:
            try:
                check, final_logs = agent.post_critic()
                num_critic += 1
                print("\n\n\n\n\n\n\n\n\n\n")
                print(final_logs)
            except Exception as e:
                print("Error in Critic")
                print(e)
                break
