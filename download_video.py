#!/usr/bin/env python3
"""
YouTube Video Downloader Script

Downloads a video from YouTube to videos/<video-id>/file.mp4

Dependencies (install in venv):
    pip install yt-dlp

Usage:
    python download_video.py <youtube_url>
    python download_video.py <youtube_url> --cookies cookies.txt

Examples:
    python download_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python download_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies cookies.txt
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def download_video(url: str, cookies_file: str = None) -> str:
    """
    Download YouTube video to videos/<video-id>/file.mp4 using yt-dlp.
    
    Args:
        url: YouTube video URL
        cookies_file: Optional path to Netscape format cookies file
        
    Returns:
        Path to downloaded video file
    """
    video_id = extract_video_id(url)
    output_dir = Path("videos") / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "file.mp4"
    
    print(f"Video ID: {video_id}")
    print(f"Output directory: {output_dir}")
    
    # Build yt-dlp command
    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        "--progress",
    ]
    
    # Add cookies if provided
    if cookies_file:
        cookies_path = Path(cookies_file)
        if cookies_path.exists():
            print(f"Using cookies from: {cookies_file}")
            cmd.extend(["--cookies", str(cookies_path)])
        else:
            print(f"Warning: Cookies file not found: {cookies_file}")
    
    cmd.append(url)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\nSuccess! Video saved to: {output_path} ({size_mb:.1f} MB)")
            return str(output_path)
        else:
            raise Exception("Download completed but file not found")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video to videos/<video-id>/file.mp4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--cookies", "-c",
        help="Path to Netscape format cookies file (export from browser with extension)",
        default=None
    )
    
    args = parser.parse_args()
    download_video(args.url, args.cookies)


if __name__ == "__main__":
    main()

