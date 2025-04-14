import asyncio
from fastapi.responses import StreamingResponse
import os
from fastapi import FastAPI, Form, Header, HTTPException, Request
from autogen_agentchat.messages import ModelClientStreamingChunkEvent
from autogen_agentchat.base import TaskResult
from fastapi.responses import StreamingResponse, JSONResponse
import base64
import uvicorn
import os
from video_pipeline.utilities.file_validator_and_remover import (
    check_if_video_processed,
    load_required_files,
    remove_file,
)
from video_pipeline.utilities.file_validator_and_remover import (
    get_frames,
    upload_gif_to_blob,
)
from video_pipeline.utilities.timestamp_classification import (
    classify_final_answer,
)
from video_pipeline.utilities.accumulate_final_answer import (
    generate_final_answer,
)
from image_pipeline.agents.main import main as image_main
from video_pipeline.agents.main import main
from video_pipeline.cache import CacheSystem
import uuid
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from PIL import Image
from video_search_pipeline.video_search import query_search
import aiofiles
from fastapi import FastAPI, Form, File, UploadFile
from utils import (
    translate_query,
    get_jwt_token,
    generate_sha256_hash,
    verify_sha256_hash,
    token_required,
)
from custom_logger import logger
from dotenv import load_dotenv

load_dotenv(override=True)
from io import BytesIO
import ast
import json

app = FastAPI()


# Async helper function to decode a base64-encoded image and save it
async def decode_and_save_image(base64_str: str, upload_folder: str = "uploads") -> str:
    """
    Asynchronously decodes a base64-encoded string and saves it as a local file.
    Returns the file path of the stored image.
    """
    # Ensure the uploads folder exists using a thread to avoid blocking
    await asyncio.to_thread(os.makedirs, upload_folder, exist_ok=True)

    # Decode the base64 image
    image_data = base64.b64decode(base64_str)

    # Generate a static-ish filename with UUID
    filename = f"uploaded_image_{str(uuid.uuid1()).split('-')[0]}.png"
    file_path = os.path.join(upload_folder, filename)

    # Write the decoded image using aiofiles for async file I/O
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(image_data)

    return file_path


# # Stream stdout during the execution of main.py
# async def stream_stdout(
#     image_path: str, query: str, tools: str, criticFlag, request_id
# ):
#     """
#     Runs the main function and streams stdout output in real-time.
#     """
#     try:
#         file_path = "image_pipeline/agents/main.py"
#         if os.path.exists(file_path) and os.path.isfile(file_path):
#             print(f"The file '{file_path}' exists.")
#         else:
#             print(f"The file '{file_path}' does not exist.")

#         logger.info(f"{request_id}: streaming started for image pipeline")
#         process = await asyncio.create_subprocess_exec(
#             "python",
#             "image_pipeline/agents/main.py",
#             image_path,
#             query,
#             tools,
#             str(criticFlag),  # Replace with your script arguments
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE,
#         )

#         # Read stdout line by line and stream it
#         assert process.stdout is not None
#         while True:
#             line = await process.stdout.readline()
#             if not line:
#                 break
#             yield line

#         # Ensure the process has completed
#         await process.wait()
#     except Exception as e:
#         yield f"Error: {str(e)}"
#         logger.error(f"{request_id}: Error occured : {str(e)}")
#     finally:
#         logger.info(f"{request_id}: streaming finished successfully")
#         if "image_path" in locals() and os.path.exists(image_path):
#             os.remove(image_path)


# async def stream_video_stdout(
#     session_id: str, query: str, tools: str, criticFlag: str, debug: str, request_id
# ):
#     """
#     Runs the main function in an isolated subprocess and streams stdout output in real-time.
#     """
#     try:
#         file_path = "video_pipeline/agents/main.py"

#         if os.path.exists(file_path) and os.path.isfile(file_path):
#             print(f"The file '{file_path}' exists.")
#         else:
#             print(f"The file '{file_path}' does not exist.")
#             raise FileNotFoundError(
#                 f"Streaming function is not accessible, Kindly check with the folder structure."
#             )
#         logger.info(f"{request_id}: streaming started")
#         process = await asyncio.create_subprocess_exec(
#             "python",
#             "video_pipeline/agents/main.py",
#             session_id,
#             query,
#             tools,
#             criticFlag,
#             debug,  # Replace with your script arguments
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE,
#             limit=1024 * 1024,
#         )

#         # Read stdout line by line and stream it
#         assert process.stdout is not None
#         while True:
#             line = await process.stdout.readline()
#             if not line:
#                 break
#             yield line

#         # Ensure the process has completed
#         await process.wait()

#         if process.returncode != 0:
#             stderr = await process.stderr.read()
#             raise RuntimeError(f"Error occured : {stderr.decode('utf-8')}")
#     except Exception as e:
#         logger.error(f"{request_id}: error occured : {str(e)}")
#         yield f"Error: {str(e)}"
#         raise
#     finally:
#         logger.info(f"{request_id}: streaming finished successfully")
#         # remove_file(session_id=session_id)
#         print("files deleted")


@app.get(
    "/",
    summary="Home Endpoint",
    description="Returns a health check message to indicate the API is running",
    response_description="A JSON object containing a health check message",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {"example": {"message": "API is runinng!"}}
            },
        }
    },
)
def home():
    """
    Home Endpoint

    This endpoint serves as a basic health check message.
    """
    return JSONResponse(content={"message": "API is runinng!"}, status_code=200)


# with open("cred.json", "r") as f:
#     creds = json.load(f)


# @app.post(
#     "/login",
#     summary="Endpoint to login to the application",
#     description="This endpoint validates the login parameters"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "email: <valid_email>\n"
#     "password : <password>\n"
#     "```",
# )
# def login(email: str = Form(...), password: str = Form(...)):
#     email_hash = generate_sha256_hash(email)
#     if email_hash in creds.keys():
#         if verify_sha256_hash(input_string=password, hash_to_check=creds[email_hash]):
#             return JSONResponse(
#                 content={
#                     "message": "Successfully logged in!",
#                     "token": get_jwt_token(email),
#                 }
#             )
#         else:
#             return JSONResponse(
#                 content={"message": "Incorrect Password. Try Again!"}, status_code=401
#             )
#     else:
#         return JSONResponse(
#             content={"message": "Email does not exist"}, status_code=401
#         )


# @app.post(
#     "/check_processed",
#     summary="Check Processed Endpoint",
#     description="This endpoint checks whether a file is already processed or not in MMCT Autogen Video Pipeline."
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tsession_id: 0XatxW6kVXY\n"
#     "```",
#     response_description="A Json object containing a boolean message.",
#     responses={
#         200: {
#             "description": "Successful Response",
#             "content": {"application/json": {"example": {"message": True}}},
#         }
#     },
# )
# @token_required
# async def check_if_video_processed_or_not(
#     session_id: str = Form(
#         ...,
#         title="session id",
#         description="A unique identifier generated using a sha256 of video content or unique id of the youtube url",
#     ),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         return JSONResponse(
#             content={"message": check_if_video_processed(session_id=session_id)},
#             status_code=200,
#         )
#     except Exception as e:
#         return JSONResponse(content={"message": str(e)}, status_code=400)


# @app.post(
#     "/stream-image-output",
#     summary="Endpoint to Stream MMCT Autogen Image output",
#     description="This endpoint streams the autogen output of image pipeline"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\timage_base64: base64_encoded_string_of_image\n"
#     "\tquery : what is shown in the image?\n"
#     "\ttools : QueryRECOGTool,QueryVITTool,QueryOCRTool,QueryObjectDetectTool\n"
#     "\tcriticFlag : True\n"
#     "```",
# )
# @token_required
# async def stream_output(
#     image_base64: str = Form(..., description="Base64 string of image"),
#     query: str = Form(..., description="User Query"),
#     tools: str = Form(..., description="String containing tool names"),
#     criticFlag: bool = Form(
#         ..., title="Critic Flag", description="Whether to use Critic Agent or not"
#     ),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         request_id = uuid.uuid1().__str__().split("-")[0]
#         logger.info(f"{request_id}: streaming request received for image pipeline")
#         logger.info(f"{request_id}: query: {query}")
#         logger.info(f"{request_id}: criticFlag: {criticFlag}")
#         logger.info(f"{request_id}: tools selected: {tools}")
#         # Decode and save the base64-encoded image; overwrite previous file if it exists
#         image_path = await decode_and_save_image(image_base64)
#         logger.info(f"{request_id}: image created from base64 string")
#         updated_query = await translate_query(text=query)
#         logger.info(f"{request_id}: query updated through translation")
#         # Pass the image path to the streaming function
#         return StreamingResponse(
#             stream_stdout(
#                 image_path=image_path,
#                 query=updated_query,
#                 tools=tools,
#                 criticFlag=criticFlag,
#                 request_id=request_id,
#             ),
#             media_type="text/plain",
#         )
#     except Exception as e:
#         logger.error(f"{request_id}: Error occured : {str(e)}")
#         return JSONResponse(content={"message": str(e)}, status_code=400)


# @app.post(
#     "/video-final-output",
#     summary="Endpoint for the final MMCT Autogen Video output",
#     description="This endpoint directly provide the MMCT video pipeline response without streaming"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tsession_id: 0XatxW6kVXY\n"
#     "\tquery : How tall can the IPA 15-6 variety of pigeon pea plants grow?\n"
#     "\ttools : query_frames_Azure_Computer_Vision,query_transcript,query_GPT4_Vision,get_transcript\n"
#     "\tcriticFlag : True\n"
#     "\tdebug: False\n"
#     "```",
# )
# @token_required
# async def final_video_output(
#     session_id: str,
#     query: str,
#     criticFlag: str = "True",
#     tools_str: str = "get_transcript,query_GPT4_Vision,query_transcript,query_frames_Azure_Computer_Vision",
#     debug: bool = False,
# ):
#     try:
#         await load_required_files(session_id=session_id)
#         result = await main(
#             session_id=session_id,
#             query=query,
#             criticFlag=criticFlag,
#             tools_str=tools_str,
#             debug=debug,
#         )
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/stream-image-qna",
    summary="Endpoint to Stream MMCT Autogen Image output",
    description="This endpoint streams the autogen output of image pipeline"
    "\n### Example Payload:\n\n"
    "```\n"
    "Header:\n"
    "\tAuthorization: <valid jwt_token in headers>\n\n"
    "Request Body:\n"
    "\timage_base64: base64_encoded_string_of_image\n"
    "\tquery : what is shown in the image?\n"
    "\ttools : QueryRECOGTool,QueryVITTool,QueryOCRTool,QueryObjectDetectTool\n"
    "\tcriticFlag : True\n"
    "```",
)
@token_required
async def stream_image_qna(
    image_base64: str = Form(..., description="Base64 string of image"),
    query: str = Form(..., description="User Query"),
    tools: str = Form(..., description="String containing tool names"),
    criticFlag: str = Form(
        ..., title="Critic Flag", description="Whether to use Critic Agent or not"
    ),
    Authorization: str = Header(
        ..., description="Authorization should contain the valid jwt token"
    ),
    ):
    try:
        image_path = await decode_and_save_image(image_base64)
        async def event_generator():
            async for message in await image_main(
                image_path=image_path,
                query=query,
                tools=tools,
                criticFlag=criticFlag,
                stream=True
                ):
                try:
                    if isinstance(message,TaskResult):
                        continue
                    if not isinstance(message,ModelClientStreamingChunkEvent):
                        data = {
                            "content": [{'arguments': cntent.arguments, 'name': cntent.name} if message.type=='ToolCallRequestEvent' else {'content': cntent.content,'name': cntent.name, 'is_error': cntent.is_error} for cntent in message.content] if isinstance(message.content,list) else message.content,
                            "source": message.source,
                            # "models_usage": message.models_usage,
                            "metadata": message.metadata,
                            "type": message.type
                        }
                        yield json.dumps(data)  + "\n" # Send raw JSON without "data: "
                except Exception as e:
                    yield f"Exception: {e}"
        
        return StreamingResponse(event_generator(), media_type="application/json")
    except Exception as e:
        HTTPException(status_code=500, detail=str(e))
    finally:
        if "image_path" in locals() and os.path.exists(image_path):
            os.remove(image_path)

@app.post(
    "/stream-video-qna",
    summary="Endpoint to stream MMCT Video QnA response",
    description="This endpoint streams real-time answers to queries based on an ingested video. Users provide a query and a video ID (which is already processed in the pipeline), and the system returns relevant responses. Powered by MMCT."
    "\n### Example Payload:\n\n"
    "```\n"
    "Header:\n"
    "\tAuthorization: <valid jwt_token in headers>\n\n"
    "Request Body:\n"
    "\tsession_id: 0XatxW6kVXY\n"
    "\tquery : How tall can the IPA 15-6 variety of pigeon pea plants grow?\n"
    "\tcriticFlag : True\n"
    "```",
)
async def stream_video_qna(
    query: str = Form(
        ..., description="Query to be answered based on the video content."
    ),
    video_id: str = Form(
        ..., description="ID of the video from which the answer is extracted."
    ),
    critic_flag: bool = Form(
        ..., description="Flag to indicate whether to use the critic agent during QnA."
    ),
    ):
    try:
        query = await translate_query(text=query)
        await load_required_files(session_id=video_id)
        print("loaded required files")
        async def event_generator():
            async for message in await main(video_id=video_id, query=query, criticFlag=critic_flag, stream=True):
                try:
                    if isinstance(message,TaskResult):
                        continue
                    if not isinstance(message,ModelClientStreamingChunkEvent):
                        data = {
                            "content": [{'arguments': cntent.arguments, 'name': cntent.name} if message.type=='ToolCallRequestEvent' else {'content': cntent.content,'name': cntent.name, 'is_error': cntent.is_error} for cntent in message.content] if isinstance(message.content,list) else message.content,
                            "source": message.source,
                            # "models_usage": message.models_usage,
                            "metadata": message.metadata,
                            "type": message.type
                        }
                        yield json.dumps(data)  + "\n" # Send raw JSON without "data: "
                except Exception as e:
                    yield f"Exception: {e}"
        
        return StreamingResponse(event_generator(), media_type="application/json")
    except Exception as e:
        HTTPException(status_code=500, detail=str(e))
    finally:
        await remove_file(session_id=video_id)

# @app.post(
#     "/video-qna",
#     summary="Endpoint to get response from MMCT on VQnA task",
#     description="This endpoint provides the response from the MMCT on video question & answering task. Users provide a query and a video ID (which is already processed in the pipeline), and the system returns relevant responses. Powered by MMCT."
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tsession_id: 0XatxW6kVXY\n"
#     "\tquery : How tall can the IPA 15-6 variety of pigeon pea plants grow?\n"
#     "\tcriticFlag : True\n"
#     "```",
# )
# async def video_qna(
#     query: str = Form(
#         ..., description="Query to be answered based on the video content."
#     ),
#     video_id: str = Form(
#         ..., description="ID of the video from which the answer is extracted."
#     ),
#     critic_flag: bool = Form(
#         ..., description="Flag to indicate whether to use the critic agent during QnA."
#     ),
#     ):
#     try:
#         async def fetch():
#             return await main(video_id=video_id, query=query, criticFlag=critic_flag, stream=False)

#         response = asyncio.run(fetch())  # Run the async function properly
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
        
# @app.post(
#     "/stream-video-output",
#     summary="Endpoint to Stream MMCT Autogen Video output",
#     description="This endpoint streams the MMCT autogen output of video pipeline"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tsession_id: 0XatxW6kVXY\n"
#     "\tquery : How tall can the IPA 15-6 variety of pigeon pea plants grow?\n"
#     "\ttools : query_frames_Azure_Computer_Vision,query_transcript,query_GPT4_Vision,get_transcript\n"
#     "\tcriticFlag : True\n"
#     "\tdebug: False\n"
#     "```",
# )
# @token_required
# async def stream_video(
#     session_id: str = Form(..., description="session id"),
#     query: str = Form(..., description="User Input Query"),
#     tools: str = Form(..., description="Selected Tools of Planner"),
#     criticFlag: str = Form(..., description="Flag whether to use critic tool or not"),
#     debug: str = Form(..., description="Enable monitoring tool"),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         request_id = uuid.uuid1().__str__().split("-")[0]
#         logger.info(f"{request_id}: streaming request received for video pipeline")
#         await load_required_files(session_id=session_id)
#         logger.info(f"{request_id}: timestamps, frames and summary loaded")
#         query = await translate_query(text=query)
#         logger.info(f"{request_id}: translated the query")
#         return StreamingResponse(
#             stream_video_stdout(
#                 session_id=session_id,
#                 query=query,
#                 tools=tools,
#                 criticFlag=criticFlag,
#                 debug=debug,
#                 request_id=request_id,
#             ),
#             media_type="text/plain",
#         )
#     except Exception as e:
#         logger.error(f"{request_id}: error occured : {str(e)}")
#         return JSONResponse(content={"message": str(e)}, status_code=400)


# @app.post(
#     "/video-search",
#     summary="Video AI Search Endpoint",
#     description="Endpoint to retrieve youtube urls on the basis of input query"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tquery : what is shown in the image?\n"
#     "\tindex_name : bihar-video-index-v2 | jharkhand-video-index-v2 | telugu-video-index-v2 | odiya-video-index-v2 \n"
#     "\ttop_k : 3\n"
#     "```",
# )
# async def search_items(
#     query: str = Form(..., description="Input Query"),
#     index_name: str = Form(
#         ..., description="Index Name from which youtube urls will be fetched"
#     ),
#     top_k: int = Form(..., description="Top K urls to fetch"),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
#     min_threshold: int = Form(..., description="Threshold value for search"),
# ):
#     try:
#         query = await translate_query(text=query)
#         results, scores, url_ids = query_search(
#             query=query, index_name=index_name, top_n=top_k, min_threshold=min_threshold
#         )
#         # print("SCoRE: ", scores)
#         return JSONResponse(
#             content={"url": results, "url_ids": url_ids}, status_code=200
#         )
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)


# @app.post(
#     "/extract_timestamp",
#     summary="Timestamp Extraction Endpoint",
#     description="Endpoint to extract timestamp from the final answer"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tfinal_answer : Process of creating Jeevamrit can be found at 2:52 | (any final answer given by the MMCT AutoGen Agents)\n"
#     "```",
# )
# @token_required
# async def extract_timestamp(
#     final_answer: str = Form(..., description="Final Answer of the Video pipeline"),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         return JSONResponse(
#             content=classify_final_answer(input_string=final_answer), status_code=200
#         )
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)


# @app.post(
#     "/generate_final_response",
#     summary="Final Answer Generation Endpoint",
#     description="Endpoint to generate final answer from multiple answers generated from multiple sources"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tquery : Illustrate the process of creating Jeevamrit.\n"
#     "\tanswers : 'Process of creating Jeevamrit can be found at 2:52\n\n\tProcess of Jeevamrit cannot be found in the provided video and context.'\n"
#     "```",
# )
# @token_required
# async def generate_final_response(
#     query: str = Form(..., description="User Query"),
#     answers: list = Form(
#         ..., description="Multiple answers generated from different multiple sources"
#     ),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         return JSONResponse(
#             content=generate_final_answer(user_query=query, answers=answers),
#             status_code=200,
#         )
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)


# def save_gif(base64_images, duration=500, loop=0):
#     frames = []
#     for b64_str in base64_images:
#         image_data = base64.b64decode(b64_str)
#         image = Image.open(BytesIO(image_data))
#         frames.append(image)

#     filename = "output.gif"
#     if frames:
#         frames[0].save(
#             filename,
#             save_all=True,
#             append_images=frames[1:],
#             duration=duration,
#             loop=loop,
#             format="GIF",
#         )
#         print("GIF created")
#         return filename
#     else:
#         print("No gif created")


# @app.post(
#     "/creategif",
#     summary="GIF Creation Endpoint",
#     description="To create gif from the given timestamps of different video ids"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\ttimestamp_details: '[{session_id: xyz,timestamp:[4,25,140]}]'\n"
#     "```"
#     "\nTimestamp are in seconds",
# )
# @token_required
# async def create_gif(
#     timestamp_details: str = Form(...),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         blob_service_client = BlobServiceClient(
#             os.getenv("BLOB_ACCOUNT_URL"), DefaultAzureCredential()
#         )
#         # print(timestamp_details)
#         timestamp_details = ast.literal_eval(timestamp_details)
#         session_ids = [i["session_id"] for i in timestamp_details]
#         frames = await get_frames(session_ids=session_ids)
#         gif_b64_strings = []
#         for idx, block in enumerate(timestamp_details):
#             for tstamp in block["timestamp"]:
#                 st_stamp = tstamp - 3 if tstamp - 3 > 0 else tstamp
#                 gif_b64_strings.extend(frames[idx][st_stamp : tstamp + 3])

#         filename = save_gif(base64_images=gif_b64_strings)
#         upload_gif_to_blob(
#             blob_service_client=blob_service_client,
#             output_path=filename,
#             blob_name=filename,
#         )
#         return JSONResponse(content={"blob_name": filename})
#     except Exception as e:
#         return JSONResponse(content={"message": f"Error occured {e}"}, status_code=400)


# class CacheSystemWrapper:
#     """Helper to handle async initialization of CacheSystem."""

#     @classmethod
#     async def create(cls, index_name, cache_probability):
#         """Proper async initializer for CacheSystem."""
#         instance = CacheSystem.__new__(CacheSystem)
#         await instance.__init__(index_name, cache_probability)  # Call async init
#         return instance


# @app.post(
#     "/cache_based_video_pipeline",
#     summary="cache based MMCT video pipeline",
#     description="Route to retrieve answer from MMCT pipeline with Cache Support"
#     "\n### Example Payload:\n\n"
#     "```\n"
#     "Header:\n"
#     "\tAuthorization: <valid jwt_token in headers>\n\n"
#     "Request Body:\n"
#     "\tquery : <user query>\n"
#     "\tregion: Jharkhand | Bihar | Odisha | Telangana\n"
#     "\tai_search_index_name : bihar-video-index-v2 | jharkhand-video-index-v2 | telugu-video-index-v2 | odiya-video-index-v2 | swahili-video-index-v2 | general-video-index-v2\n"
#     "\ttop_k : 3 [top k documents to be selected\n"
#     "\tmin_threshold: 80 [threshold above which documents from AI search should get selected]"
#     "```",
# )
# @token_required
# async def cache_based_video_pipeline(
#     query: str = Form(..., description="Query"),
#     region: str = Form(..., description="Region to which query belongs to."),
#     top_k: str = Form(
#         ..., description="Top k video ids to retrieve from the AI Search"
#     ),
#     min_threshold: str = Form(..., description="minimum threshold for search score"),
#     ai_search_index_name: str = Form(..., description="Azure AI search index name"),
#     Authorization: str = Header(
#         ..., description="Authorization should contain the valid jwt token"
#     ),
# ):
#     try:
#         print(f"This is query:{query}")
#         CACHE_INDEX_NAME = os.environ.get("CACHE_INDEX_NAME")
#         # Checking query in cache
#         cache_system = await CacheSystemWrapper.create(
#             index_name=CACHE_INDEX_NAME, cache_probability=1
#         )  # 100% probability of checking cache
#         cache_result = await cache_system.get_data(query=query, region=region)

#         print("This is cache_result:", cache_result)
#         print(type(cache_result["cache_hit"]))
#         if cache_result.get("cache_hit") == True:
#             print("here")
#             return JSONResponse(content=cache_result, status_code=200)

#         # translating query into english
#         query = await translate_query(query)

#         # Fetching video_ids related to query
#         _, _, url_ids = query_search(
#             query=query,
#             index_name=ai_search_index_name,
#             top_n=int(top_k),
#             min_threshold=float(min_threshold),
#         )
#         print(url_ids)

#         # Load required files like transcript, frames, etc.
#         await asyncio.gather(
#             *(load_required_files(session_id=session_id) for session_id in url_ids)
#         )
#         print("loaded required data")

#         # Run MMCT for each video
#         async def fetch_result(session_id):
#             return session_id, await main(session_id=session_id, query=query)

#         session_results = await asyncio.gather(
#             *(fetch_result(session_id) for session_id in url_ids)
#         )
#         print(f"Generated MMCT response:{session_results}")

#         # 4. Store responses in a dictionary
#         response_dict = response_dict = {
#             session_id: data["content"] for session_id, data in session_results
#         }
#         print(response_dict)

#         final_response = generate_final_answer(
#             user_query=query, response_dict=response_dict, cache_system=True
#         )
#         final_response["cache_hit"] = False
#         print(f"Generated final answer:{final_response}")
#         if final_response.get("answer_found") == "True":
#             await cache_system.insertDocumentInCache(
#                 question=query,
#                 answer=final_response["response"],
#                 region=region,
#                 video_id_list=final_response["source"],
#             )

#             return final_response
#         else:
#             return {
#                 "cache_hit": False,
#                 "source": [],
#                 "response": "I couldn't find an answer. Try rephrasing or providing more details.",
#             }
#     except Exception as e:
#         return JSONResponse(status_code=400, content=str(e))


if __name__ == "__main__":
    logger.info("backend is running")
    uvicorn.run(app, host="0.0.0.0", port=8000)
