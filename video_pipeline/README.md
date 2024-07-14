
# MMCT VIDEO

## Setup Instructions

### Install Dependencies
To install required dependencies, run:
```bash
pip install -r requirements.txt
```
Apart from this you need to install FFmpeg

### Configure Environment Variables
Configure the necessary environment variables using the provided script:
```bash
chmod +x secrets_mmct_video.sh
./secrets_mmct_video.sh
```

## Usage

Run the VideoAgent using the following command structure:
```bash
python mmct_video.py <video_path> "<question>" [--critic] [--max_num_critic <num>]
```

### Examples

**Example 1: Basic Usage**
Answer a question about a video without using the critic:
```bash
python mmct_video.py /path/to/video.mp4 "What objects are visible in the first scene?"
```

**Example 2: Using the Critic**
To use the critic and further analyze the query and refine responses, add the `--critic` flag and optionally specify the maximum number of critic iterations:
```bash
python mmct_video.py /path/to/video.mp4 "Who is speaking at the beginning of the video?" --critic --max_num_critic 3
```
