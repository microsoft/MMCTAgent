
# MMCT VIDEO

## Setup Instructions

### Install Dependencies
To install required dependencies, run:
```bash
pip install -r requirements.txt
```
Apart from this you need to install FFmpeg

### Setup Azure Resources
The system requires the following Azure Resources:

1) Blob Storage (https://azure.microsoft.com/en-in/products/storage/blobs/)
2) Azure AI Vision (https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/video-retrieval#prerequisites)
3) Azure Open AI with gpt-4-32k, gpt-4-vision-preview, text-embedding-ada-002 and whisper (https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models)
4) Azure AI Content Safety (https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety/)


### Configure Environment Variables
Configure the necessary environment variables using the provided script:
```bash
chmod +x secrets_mmct_video.sh
./secrets_mmct_video.sh
```

## Usage

Run the VideoAgent using the following command structure:
```bash
python mmct_video.py <video_path> "<question>" [--critic] [--max_num_critic <num>] [--moderation <bool>]
```

### Arguments

<video_path>: Path to the video file you want to analyze.
"<question>": The question you want to ask about the video (enclose in quotes).
--critic: (Optional) Flag to use the critic for further analysis and refinement of responses.
--max_num_critic <num>: (Optional) Maximum number of critic iterations (default is 1).
--moderation <bool>: (Optional) Whether to use content moderation (default is True).

### Examples

**Example 1: Basic Usage**
Answer a question about a video without using the critic:
```bash
python mmct_video.py /path/to/video.mp4 "Describe the scene in detail"
```

**Example 2: Using the Critic**
To use the critic and further analyze the query and refine responses, add the `--critic` flag and optionally specify the maximum number of critic iterations:
```bash
python mmct_video.py /path/to/video.mp4 "Describe the scene in detail" --critic --max_num_critic 3
```

**Example 3: Disable Content Moderation**
To disable content moderation, set the --moderation flag to False:
```bash
python mmct_video.py /path/to/video.mp4 "Describe the scene in detail" --moderation False
```
Note: Content moderation is enabled by default to filter inappropriate content. Disable it only when you're sure about the content of your video and questions.
