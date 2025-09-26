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
import math
from io import BytesIO
from datetime import timedelta
from typing import Dict
from azure.storage.blob import BlobServiceClient
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from mmct.video_pipeline.utils.ai_search_client import AISearchClient
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


async def download_blobs(blob_names, output_dir, container_name=None):
    if container_name is None:
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

    download_tasks = [download_single(blob_name) for blob_name in blob_names]
    results = await asyncio.gather(
        *download_tasks
    )
    # results = await asyncio.gather(
    #     *(download_single(blob_name) for blob_name in blob_names)
    # )
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


async def stack_images_in_grid(frames, grid_size=4, type="base64"):
    """
    Stack images in a grid format (e.g., 2x2 for grid_size=4, 3x3 for grid_size=9).
    
    Args:
        frames: List of base64 encoded images or PIL images
        grid_size: Number of images per grid (4, 9, 16, etc.)
        type: "base64" or "pil" depending on input format
    
    Returns:
        List of PIL images representing grids
    """
    try:
        if not frames:
            return []
        
        # Convert to PIL images if needed
        if type == "base64":
            images = await asyncio.gather(
                *[decode_base64_to_image(img) for img in frames]
            )
        else:
            images = frames
        
        # Calculate grid dimensions (square root for square grid)
        grid_cols = int(math.sqrt(grid_size))
        grid_rows = int(math.ceil(grid_size / grid_cols))
        
        # Group frames into batches
        grids = []
        for i in range(0, len(images), grid_size):
            batch = images[i:i + grid_size]
            
            if not batch:
                continue
                
            # Get dimensions for uniform sizing
            max_width = max(img.width for img in batch)
            max_height = max(img.height for img in batch)
            
            # Create grid canvas
            canvas_width = max_width * grid_cols
            canvas_height = max_height * grid_rows
            grid_image = Image.new("RGB", (canvas_width, canvas_height), color="white")
            
            # Place images in grid
            for idx, img in enumerate(batch):
                row = idx // grid_cols
                col = idx % grid_cols
                
                # Resize image to fit grid cell while maintaining aspect ratio
                img_resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                
                x_pos = col * max_width
                y_pos = row * max_height
                grid_image.paste(img_resized, (x_pos, y_pos))
            
            grids.append(grid_image)
        
        return grids
    except Exception as e:
        raise Exception(f"Error while stacking images in grid: {e}")


async def create_stacked_frames_base64(frames, grid_size=4, enable_stacking=True):
    """
    Create stacked frames in base64 format for LLM processing.
    
    Args:
        frames: List of base64 encoded images
        grid_size: Number of images per grid (default: 4 for 2x2)
        enable_stacking: Whether to enable frame stacking (default: True)
    
    Returns:
        List of base64 encoded stacked images or original frames
    """
    try:
        if not enable_stacking or len(frames) <= grid_size:
            return frames
        
        # Create grid images
        grid_images = await stack_images_in_grid(frames, grid_size=grid_size, type="base64")
        
        # Convert back to base64
        stacked_frames = []
        for grid_img in grid_images:
            base64_str = await encode_image_to_base64(grid_img)
            stacked_frames.append(base64_str)
        
        logger.info(f"Stacked {len(frames)} frames into {len(stacked_frames)} grid images (grid_size={grid_size})")
        return stacked_frames
    except Exception as e:
        logger.error(f"Error creating stacked frames: {e}")
        # Fallback to original frames on error
        return frames


async def load_required_files(session_id):
    try:
        base_dir = await get_media_folder()
        os.makedirs(
            base_dir,
            exist_ok=True,
        )

        def save_content(container_name, blob_name, list_flag=True, binary_data=False):
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            local_path = os.path.join(base_dir, blob_name)
            # Download the blob's content
            blob_data = blob_client.download_blob().read()

            if binary_data:
                # For binary content like video
                with open(local_path, "wb") as f:
                    f.write(blob_data)
            else:
                if list_flag == True:
                    # Decode the byte data to string and split into lines
                    content = blob_data.decode("utf-8").splitlines()
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(content))
                else:
                    content = blob_data.decode("utf-8")
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(content)

        video_blob_name = f"{session_id}.mp4"
        timestamps_blob_name = f"timestamps_{session_id}.txt"
        # frames_blob_name = f"frames_{session_id}.txt"
        summary_blob_name = f"{session_id}.json"

        # Use Azure CLI credential if available, fallback to DefaultAzureCredential
        credential = _get_credential()
            
        blob_service_client = BlobServiceClient(
            os.getenv("BLOB_ACCOUNT_URL"), credential
        )
        save_content(
            container_name=os.getenv("VIDEO_CONTAINER_NAME"),
            blob_name=video_blob_name,
            binary_data=True
        )
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


async def get_file_hash(file_path, hash_algorithm="sha256", suffix=""):
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

        hash_id = hash_func.hexdigest() + suffix
        logger.info(f"Hash Id Generated: {hash_id}")
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


async def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.
    Args:
        video_path (str): Path to the video file.
    Returns:
        float: Video duration in seconds.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting video duration: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error parsing video duration: {e}")
        raise


async def split_video_if_needed(video_path: str) -> tuple[list[str], list[str]]:
    """
    Split video into two parts if duration >= 30 minutes.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        tuple: (list of video paths, list of corresponding hash suffixes)
               - If split: ([part_A_path, part_B_path], ['', 'B'])
               - If not split: ([original_path], [''])
    """
    try:
        duration = await get_video_duration(video_path)
        
        # Check if video is >= 30 minutes (1800 seconds)
        if duration < 1800:
            logger.info("Video duration is less than 30 minutes, no splitting needed")
            return [video_path], ['']
        
        logger.info("Video duration is >= 30 minutes, splitting video into two parts")
        
        # Get video file info
        video_name, video_ext = os.path.splitext(os.path.basename(video_path))
        
        # Get media folder for output paths
        media_folder = await get_media_folder()
        
        # Calculate split point (half duration)
        split_point = duration / 2
        
        # Define output paths in media folder
        part_a_path = os.path.join(media_folder, f"{video_name}_part_A{video_ext}")
        part_b_path = os.path.join(media_folder, f"{video_name}_part_B{video_ext}")
        
        # Split video using ffmpeg
        # Part A: from start to middle
        cmd_a = [
            'ffmpeg', '-i', video_path, '-t', str(split_point), 
            '-c', 'copy', '-avoid_negative_ts', 'make_zero', part_a_path, '-y'
        ]
        
        # Part B: from middle to end
        cmd_b = [
            'ffmpeg', '-i', video_path, '-ss', str(split_point), 
            '-c', 'copy', '-avoid_negative_ts', 'make_zero', part_b_path, '-y'
        ]
        
        logger.info("Splitting video into Part A...")
        result_a = subprocess.run(cmd_a, capture_output=True, text=True, check=True)
        logger.info(f"Part A created successfully: {part_a_path}")
        
        logger.info("Splitting video into Part B...")
        result_b = subprocess.run(cmd_b, capture_output=True, text=True, check=True)
        logger.info(f"Part B created successfully: {part_b_path}")
        
        # Verify both parts were created
        if not os.path.exists(part_a_path) or not os.path.exists(part_b_path):
            raise RuntimeError("Video splitting failed - output files not found")
        
        logger.info(f"Video successfully split into:\n  Part A: {part_a_path}\n  Part B: {part_b_path}")
        return [part_a_path, part_b_path], ['', 'B']
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error splitting video: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during video splitting: {e}")
        raise


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
                    
        keyframes_dir_name = f"media/relevant_frames/{video_id}"
       
        #frames_dir_name = f"Frames/{video_id}"
        await remove_dir(keyframes_dir_name)
        
        print("All files and directories removed successfully!")
        
    except Exception as e:
        raise Exception(e)


async def check_video_already_ingested(hash_id: str, index_name: str) -> bool:
    """
    Check if a video with the given hash_id already exists in the search index.
    
    Args:
        hash_id (str): The hash ID of the video to check
        index_name (str): The name of the search index to check
        
    Returns:
        bool: True if video already exists, False otherwise
    """
    try:
        # Get credentials based on environment configuration
        if os.environ.get("MANAGED_IDENTITY", "FALSE").upper() == "TRUE":
            from azure.identity import AzureCliCredential, DefaultAzureCredential
            try:
                cli_credential = AzureCliCredential()
                cli_credential.get_token("https://search.azure.com/.default")
                credential = cli_credential
            except Exception:
                credential = DefaultAzureCredential()
        else:
            from azure.core.credentials import AzureKeyCredential
            key = os.getenv("SEARCH_SERVICE_KEY")
            if key is None:
                raise Exception("SEARCH_SERVICE_KEY is missing for Azure AI Search!")
            credential = AzureKeyCredential(key)
        
        # Create AI Search client
        index_client = AISearchClient(
            endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=index_name,
            credential=credential
        )
        
        # Check if document exists
        exists = await index_client.check_if_exists(hash_id=hash_id)
        await index_client.close()
        
        return exists
        
    except Exception as e:
        logger.error(f"Error checking if video already ingested: {e}")
        # In case of error, return False to proceed with ingestion
        return False

async def load_srt(path: str) -> str:
    """
    Asynchronously load the full contents of an SRT (SubRip Subtitle) transcript file,
    preserving subtitle indexes, timestamps, and text blocks.

    Args:
        path (str): Path to the .srt transcript file.

    Returns:
        str: The complete content of the SRT file as a single string, with original formatting.
    """
    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        content = await f.read()
    return content.strip()

def parse_srt_timestamps(srt_content: str) -> list:
    """
    Parse SRT content to extract timestamps and text segments.
    
    Args:
        srt_content (str): SRT file content as string
        
    Returns:
        list: List of dictionaries with 'start_time', 'end_time', 'text'
    """
    segments = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        # Extract timestamp line (second line)
        timestamp_line = lines[1]
        if '-->' not in timestamp_line:
            continue
            
        # Parse timestamps
        start_time_str, end_time_str = timestamp_line.split(' --> ')
        
        # Convert timestamp to seconds
        def timestamp_to_seconds(timestamp_str):
            timestamp_str = timestamp_str.replace(',', '.')
            h, m, s = timestamp_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        
        start_time = timestamp_to_seconds(start_time_str.strip())
        end_time = timestamp_to_seconds(end_time_str.strip())
        
        # Extract text (lines after timestamp)
        text = '\n'.join(lines[2:])
        
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': text
        })
    
    return segments

async def chunk_video_by_timestamps(video_path: str, timestamps: list, output_dir: str, hash_id: str) -> list:
    """
    Chunk video based on transcript timestamps for parallel processing.
    
    Args:
        video_path (str): Path to the video file
        timestamps (list): List of timestamp segments from parse_srt_timestamps
        output_dir (str): Directory to save video chunks
        hash_id (str): Hash ID for naming chunks
        
    Returns:
        list: List of paths to video chunk files
    """
    if not timestamps:
        logger.warning("No timestamps provided, returning original video")
        return [video_path]
    
    # Calculate mid-point for 50-50 split
    mid_index = len(timestamps) // 2
    
    # Get timestamps for two chunks
    chunk1_start = timestamps[0]['start_time']
    chunk1_end = timestamps[mid_index - 1]['end_time']
    chunk2_start = timestamps[mid_index]['start_time']
    chunk2_end = timestamps[-1]['end_time']
    
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_paths = []
    chunks = [
        (chunk1_start, chunk1_end, f"{hash_id}.mp4"),
        (chunk2_start, chunk2_end, f"{hash_id}B.mp4")
    ]
    
    for start_time, end_time, filename in chunks:
        output_path = os.path.join(output_dir, filename)
        
        # Use FFmpeg to extract video segment
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y', output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            chunk_paths.append(output_path)
            logger.info(f"Created video chunk: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create video chunk {output_path}: {e.stderr}")
            raise
    
    return chunk_paths

def split_transcript_by_segments(srt_content: str, segment_count: int = 2) -> list:
    """
    Split transcript content into segments for parallel processing.
    Resets timestamps for each chunk to start from 0 (matching video chunks created with FFmpeg).
    
    Args:
        srt_content (str): Original SRT content
        segment_count (int): Number of segments to split into (default: 2)
        
    Returns:
        list: List of SRT content strings for each segment
    """
    segments = parse_srt_timestamps(srt_content)
    
    if not segments:
        return [srt_content]
    
    # Split segments into chunks
    chunk_size = len(segments) // segment_count
    transcript_chunks = []
    
    for i in range(segment_count):
        start_idx = i * chunk_size
        if i == segment_count - 1:
            # Last chunk gets remaining segments
            end_idx = len(segments)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk_segments = segments[start_idx:end_idx]
        
        if not chunk_segments:
            continue
            
        # Calculate time offset for chunks after the first one (Part B, C, etc.)
        # Part A (i == 0) keeps original timestamps, others reset to start from 0
        if i == 0:
            # Part A: Keep original timestamps
            time_offset = 0
        else:
            # Part B and beyond: Reset timestamps to start from 0
            # This matches what FFmpeg does with -avoid_negative_ts make_zero
            time_offset = chunk_segments[0]['start_time']
        
        # Rebuild SRT format for this chunk
        chunk_srt = ""
        for j, segment in enumerate(chunk_segments, 1):
            # Convert seconds back to SRT timestamp format
            def seconds_to_timestamp(seconds):
                # Ensure non-negative seconds
                seconds = max(0, seconds)
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
            
            # Apply offset (0 for Part A, calculated offset for Part B+)
            adjusted_start_time = segment['start_time'] - time_offset
            adjusted_end_time = segment['end_time'] - time_offset
            
            start_timestamp = seconds_to_timestamp(adjusted_start_time)
            end_timestamp = seconds_to_timestamp(adjusted_end_time)
            
            chunk_srt += f"{j}\n{start_timestamp} --> {end_timestamp}\n{segment['text']}\n\n"
        
        transcript_chunks.append(chunk_srt.strip())
    
    return transcript_chunks

def seconds_to_hms(duration_seconds):
    """
    Convert duration in seconds to HH:MM:SS format string.
    
    Args:
        duration_seconds (int or float): Duration in seconds to convert
        
    Returns:
        str: Time formatted as "HH:MM:SS" with leading zeros
        
    Example:
        >>> seconds_to_hms(3661)
        '01:01:01'
        >>> seconds_to_hms(7200)
        '02:00:00'
        >>> seconds_to_hms(45)
        '00:00:45'
    """
    # Convert to integer to handle float inputs
    duration_seconds = int(duration_seconds)
    
    # Calculate hours, minutes, and seconds
    hours = duration_seconds // 3600
    remaining_seconds = duration_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    # Format with leading zeros
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _hhmmss_to_timedelta(ts: str) -> timedelta:
    h, m, s = map(int, ts.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

def _timedelta_to_hhmmss(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

async def _offset_single_source(src, duration_dict: Dict[str, str]) -> None:
    """
    Offset timestamps for one VideoSourceInfo if it is a Part-B video.
    Mutates `src.timestamps` in-place.
    """
    vid = src.video_id
    if len(vid) != 65 or not vid.endswith("B"):
        return  # Part-A or normal video → nothing to do

    base_id = vid[:-1]                       # strip trailing 'B'
    base_dur_str = duration_dict.get(base_id)
    if not base_dur_str:                     # unknown duration → skip
        return

    base_td = _hhmmss_to_timedelta(base_dur_str)
    src.timestamps = [
        _timedelta_to_hhmmss(_hhmmss_to_timedelta(t) + base_td)
        for t in src.timestamps  # ✅ Fixed: use dot notation
    ]


async def offset_all_sources_in_response(resp, duration_dict: Dict[str, str]) -> None:
    """
    Iterate over every VideoSourceInfo in `resp.content.source`
    and apply `_offset_single_source` concurrently.
    """
    await asyncio.gather(
    *(_offset_single_source(src, duration_dict) for src in resp['content'].source)
    )