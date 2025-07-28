import asyncio
from PIL import Image
import base64
import hashlib
import aiofiles
import shutil
from loguru import logger
import os
import cv2
import subprocess
from io import BytesIO
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def _get_credential():
    """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
    try:
        from azure.identity import AzureCliCredential
        # Try Azure CLI credential first
        cli_credential = AzureCliCredential()
        # Test if CLI credential works by getting a token
        cli_credential.get_token("https://storage.azure.com/.default")
        return cli_credential
    except Exception:
        return DefaultAzureCredential()


async def load_images(
    file_paths, valid_extensions={".jpg", ".jpeg", ".png", ".bmp", ".gif"}
):
    """
    This function loads the images from the directory with given file paths.
    """

    async def load_single_image(path):
        try:
            return Image.open(path).copy()
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None

    # Filter only valid image files
    valid_files = [
        path
        for path in file_paths
        if os.path.splitext(path)[1].lower() in valid_extensions
        and os.path.isfile(path)
    ]

    tasks = [load_single_image(path) for path in valid_files]
    images = await asyncio.gather(*tasks)
    return [img for img in images if img is not None]


async def download_blobs(blob_names, output_dir):
    container_name = os.getenv("FRAMES_CONTAINER_NAME")
    
    # Use Azure CLI credential if available, fallback to DefaultAzureCredential
    credential = _get_credential()
        
    blob_service_client = AsyncBlobServiceClient(
        os.getenv("BLOB_ACCOUNT_URL"), credential
    )
    container_client = blob_service_client.get_container_client(container_name)

    async def download_single(blob_name):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            stream = await blob_client.download_blob()

            download_path = os.path.join(output_dir, blob_name.split("/")[-1])
            os.makedirs(os.path.dirname(download_path), exist_ok=True)

            async with aiofiles.open(download_path, "wb") as f:
                async for chunk in stream.chunks():
                    await f.write(chunk)

            return blob_name.split("/")[-1]
        except Exception as e:
            print(f"Failed: {blob_name} - {e}")
            return None

    results = await asyncio.gather(
        *(download_single(blob_name) for blob_name in blob_names)
    )
    await blob_service_client.close()

    # Filter out any failed (None) results and return the successful downloads
    return [r for r in results if r is not None]


async def load_data_from_txt(filepath: str) -> list:
    try:
        """
        function to load data from txt file
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines()]
        return data
    except Exception as e:
        raise Exception(f"Error loading file: {e}")


# Function to decode base64 frames to image
async def decode_base64_to_image(base64_str):
    try:
        """Decode a base64 string to a PIL image."""
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        raise Exception(f"Error decoding the base64 image to image: {e}")


async def encode_image_to_base64(image):
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        raise Exception(f"Error encoding image to base64: {e}")


async def stack_images_horizontally(frames, type="base64"):
    try:
        if type == "base64":
            images = await asyncio.gather(
                *[decode_base64_to_image(img) for img in frames]
            )
        else:
            images = frames
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        new_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return new_img
    except Exception as e:
        raise Exception(f"Error while stacking the images horizontally: {e}")


async def load_required_files(session_id):
    try:
        base_dir = await get_media_folder()
        os.makedirs(
            base_dir,
            exist_ok=True,
        )

        def save_content(container_name, blob_name, list_flag=True):
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            local_path = os.path.join(base_dir, blob_name)
            # Download the blob's content
            blob_data = blob_client.download_blob().read()

            if list_flag == True:
                # Decode the byte data to string and split into lines
                content = blob_data.decode("utf-8").splitlines()
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(content))
            else:
                content = blob_data.decode("utf-8")
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)

        transcript_blob_name = f"transcript_{session_id}.srt"
        timestamps_blob_name = f"timestamps_{session_id}.txt"
        # frames_blob_name = f"frames_{session_id}.txt"
        summary_blob_name = f"{session_id}.json"

        # Use Azure CLI credential if available, fallback to DefaultAzureCredential
        credential = _get_credential()
            
        blob_service_client = BlobServiceClient(
            os.getenv("BLOB_ACCOUNT_URL"), credential
        )
        # save_content(
        #     container_name=os.getenv("TRANSCRIPT_CONTAINER_NAME"),
        #     blob_name=transcript_blob_name,
        #     list_flag=False,
        # )
        save_content(
            container_name=os.getenv("TIMESTAMPS_CONTAINER_NAME"),
            blob_name=timestamps_blob_name,
            list_flag=True,
        )

        logger.info(f"summary_blob_name:{summary_blob_name}")
        logger.info(f"container Name:{os.getenv('VIDEO_DESCRIPTION_CONTAINER_NAME')}")
        # save_content(container_name=os.getenv("FRAMES_CONTAINER_NAME"),blob_name=frames_blob_name,list_flag=True)
        save_content(
            container_name=os.getenv("VIDEO_DESCRIPTION_CONTAINER_NAME"),
            blob_name=summary_blob_name,
            list_flag=False,
        )
    except Exception as e:
        raise Exception(f"Error downloading files from azure storage:{e}")


async def get_file_hash(file_path, hash_algorithm="sha256"):
    try:
        """Generate a hash for a file asynchronously."""
        hash_func = hashlib.new(hash_algorithm)

        async with aiofiles.open(
            file_path, "rb"
        ) as file:  # Open the file asynchronously
            while True:
                chunk = await file.read(8192)  # Read chunks asynchronously
                if not chunk:
                    break
                hash_func.update(chunk)

        hash_id =  hash_func.hexdigest()
        logger.info("Hash Id Generated")
        return hash_id
    except Exception as e:
        logger.exception(f"Error generating hash id of video, error:{e}")
        raise


async def file_upload_to_blob(
    file_path: str, blob_file_name: str, container_name: str
) -> str:
    """Asynchronously uploads a file to Azure Blob Storage."""
    try:
        # Use Azure CLI credential if available, fallback to DefaultAzureCredential
        credential = _get_credential()
            
        blob_service_client = AsyncBlobServiceClient(
            os.getenv("BLOB_ACCOUNT_URL"), credential=credential
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_file_name
        )

        async with aiofiles.open(file_path, "rb") as data:
            file_bytes = await data.read()
            await blob_client.upload_blob(
                file_bytes, overwrite=True, blob_type="BlockBlob"
            )

        blob_url = f"{os.getenv('BLOB_ACCOUNT_URL')}/{container_name}/{blob_file_name}"
        await blob_service_client.close()
        return blob_url
    except Exception as e:
        raise Exception(f"Error uploading file to blob: {e}")


async def extract_wav_from_video(video_path: str, output_path: str):
    """Extracts audio from a video file and saves it as WAV using FFmpeg."""
    try:
        # Ensure output file has a .wav extension
        if not output_path.endswith(".wav"):
            output_path = os.path.splitext(output_path)[0] + ".wav"

        with open(os.devnull, "wb") as devnull:
            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    output_path,
                ],
                stdout=devnull,
                stderr=devnull,
            )

        return output_path
    except Exception as e:
        raise Exception(f"Error getting audio from video, error:{e}")


async def extract_mp3_from_video(video_path: str, output_path: str):
    """Extracts audio from a video file using FFmpeg."""
    try:
        with open(os.devnull, "wb") as devnull:
                subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-q:a",
                    "0",
                    "-map",
                    "a",
                    output_path,
                ],
                stdout=devnull,
                stderr=devnull,
            )
    except Exception as e:
        raise Exception(f"Error extracting audio from {video_path}: {e}")


async def extract_frames(video_path, fps=1):
    """
    Extracts frames from a video at a specified FPS.

    Args:
        video_path (str): Path to the video file.
        fps (int, optional): Frames per second to extract. Defaults to 1.

    Returns:
        tuple: (list of PIL Images, list of timestamps in milliseconds).
    """
    cap = cv2.VideoCapture(video_path)
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
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame_pil)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        frame_count += 1

    cap.release()
    return frames, timestamps


async def encode_frames_to_base64(frames):
    """
    Encodes a list of image frames into Base64 format.

    Args:
        frames (list): List of PIL Image objects.

    Returns:
        list: List of Base64-encoded image strings.
    """
    encoded_frames = []
    for frame in frames:
        buffer = BytesIO()
        frame.save(buffer, format="JPEG")
        encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return encoded_frames


async def fetch_frames_based_on_counts(frame_counts, image_frames, seconds_per_frame):
    """
    Fetches frames based on calculated time differences.

    Args:
        frame_counts (list): List of frame count intervals.
        image_frames (list): List of extracted frames.
        seconds_per_frame (int): Interval used for frame extraction.

    Returns:
        list: Frames grouped per transcript section.
    """
    start_index = 0
    frames_per_cluster = []

    for count in frame_counts:
        end_index = start_index + (
            int(count / seconds_per_frame) if seconds_per_frame > 1 else count
        )
        frames_per_cluster.append(image_frames[start_index:end_index])
        start_index = end_index

    return frames_per_cluster


async def save_frames_as_png(frames, directory_path, name_prefix="frame_"):
    """
    Saves a list of PIL Image frames as PNG files to a specified directory.
    Minimizes memory usage by processing one frame at a time.

    Args:
        frames (list): List of PIL Image objects.
        directory_path (str): Directory path where the PNG files will be saved.
        name_prefix (str): Prefix for PNG file names.

    Returns:
        list: List of paths to the saved PNG files.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        saved_paths = []
        logger.info(f"Saving frames to the provided directory: {directory_path}")
        for i, frame in enumerate(frames):
            # Generate file path
            file_path = os.path.join(directory_path, f"{name_prefix}_{i}.png")

            # Save the frame as PNG
            frame.save(file_path, format="PNG")
            saved_paths.append(file_path)

            # Log progress for every 10th frame to avoid excessive logging
            # if i % 10 == 0:
            #     logger.info(f"Saved frame {i} as PNG")

        logger.info(f"Saved {len(saved_paths)} frames as PNG files to {directory_path}")
        return saved_paths
    except Exception as e:
        logger.exception(f"Exception occured while saving frames: {e}")
        raise


# Helper to batch tasks
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


async def get_media_folder() -> str:
    """
    Returns the path to the Media folder in the current working directory.
    Ensures the folder exists.
    """
    media_path = os.path.join(os.getcwd(), "media")
    os.makedirs(media_path, exist_ok=True)
    return media_path

async def remove_file(video_id):
    try:
        base_dir = await get_media_folder()

        async def remove_entity(filename):
            local_path = os.path.join(base_dir, filename)
            try:
                print(f"Trying to remove file: {local_path}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                    print(f"Trying to remove file: {local_path}")
                    
            except Exception as e:
                print(f"Error deleting file {local_path}: {e}")

        async def remove_dir(dir_name):
            local_path = os.path.join(base_dir, dir_name)
            try:
                print(f"Trying to remove directory: {local_path}")
                if os.path.exists(local_path):
                    shutil.rmtree(local_path)
            except Exception as e:
                print(f"Error deleting directory {local_path}: {e}")
                    
        transcript_blob_name = f"transcript_{video_id}.srt"
        timestamps_blob_name = f"timestamps_{video_id}.txt"
        frames_dir_name = f"Frames/{video_id}"
        compressed_video_file_name = f"Compressed_Videos/{video_id}.mp4"
        summary_blob_name = f"{video_id}.json"
        audio_wav_name = f"{video_id}.wav"
        audio_mp3_name = f"{video_id}.mp3"
        await remove_entity(transcript_blob_name)
        await remove_entity(timestamps_blob_name)
        await remove_dir(frames_dir_name)
        await remove_entity(summary_blob_name)
        await remove_entity(audio_wav_name)
        await remove_entity(audio_mp3_name)
        await remove_entity(compressed_video_file_name)
        print("All files and directories removed successfully!")
        
    except Exception as e:
        raise Exception(e)
