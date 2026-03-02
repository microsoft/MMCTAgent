#!/usr/bin/env python3
"""
Download all keyframe images referenced in vision test logs and produce
a JSON manifest mapping each local image path to the ImageAgent task
(the query/directive the agent used when analyzing that image).

Usage:
    python app/download_vision_keyframes.py
    python app/download_vision_keyframes.py --log-dir app/logs_GPT_4.1_no_critic_no_reflection_vision
    python app/download_vision_keyframes.py --output-dir keyframes_downloaded
"""

import argparse
import asyncio
import json
import os
import re
import sys
import glob
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from app.config.provider_config import get_settings
from app.config.credentials import resolve_credentials
from mmct.providers.azure import AzureStorageProvider


def parse_logs(log_dir: str) -> list[dict]:
    """Parse event logs and extract ImageAgent tool calls with blob URLs and queries.

    Returns a list of dicts:
        {
            "query_index": int,
            "user_query": str,
            "blob_url": str,
            "image_agent_query": str,
            "metadata": dict,
        }
    """
    summary_files = glob.glob(os.path.join(log_dir, "test_vision_queries_*.json"))
    event_files = sorted(glob.glob(os.path.join(log_dir, "v4_query_*.json")))

    if not event_files:
        raise FileNotFoundError(f"No v4_query_*.json files found in {log_dir}")

    # Load summary for user-level query text
    user_queries = {}
    if summary_files:
        with open(summary_files[0]) as f:
            summary = json.load(f)
        for r in summary.get("results", []):
            user_queries[r["index"]] = r["query"]

    records = []
    seen_urls = set()

    for qi, fpath in enumerate(event_files):
        with open(fpath) as f:
            events = json.load(f)

        query_idx = qi + 1
        user_query = user_queries.get(query_idx, "")

        for ev in events:
            agent = ev.get("agent", "")
            content = str(ev.get("content", ""))

            if agent != "ImageAgent":
                continue
            if "analyze_image_with_vit" not in content:
                continue
            if "FunctionCall" not in content:
                continue

            # Extract all FunctionCall arguments blocks
            matches = re.findall(r"arguments='(\{[^']+\})'", content)
            for m in matches:
                try:
                    args = json.loads(m)
                except json.JSONDecodeError:
                    continue

                blob_url = args.get("image_path", "")
                if not blob_url or blob_url in seen_urls:
                    continue
                seen_urls.add(blob_url)

                records.append({
                    "query_index": query_idx,
                    "user_query": user_query,
                    "blob_url": blob_url,
                    "image_agent_query": args.get("query", ""),
                    "metadata": args.get("metadata", {}),
                })

    return records


def blob_url_to_path(blob_url: str, container_name: str) -> tuple[str, str]:
    """Extract (container, blob_path) from a full blob URL.

    Example:
        https://account.blob.core.windows.net/keyframes/vid/keyframes/1/1_kf_0000.jpg
        -> ("keyframes", "vid/keyframes/1/1_kf_0000.jpg")
    """
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")
    if path.startswith(container_name + "/"):
        blob_path = path[len(container_name) + 1:]
    else:
        blob_path = path
    return container_name, blob_path


async def download_all(
    records: list[dict],
    output_dir: str,
    storage_provider: AzureStorageProvider,
    container_name: str,
) -> list[dict]:
    """Download all keyframes and return manifest entries."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    total = len(records)

    for i, rec in enumerate(records):
        blob_url = rec["blob_url"]
        container, blob_path = blob_url_to_path(blob_url, container_name)

        # Create a safe local filename preserving video/chapter structure
        local_rel = blob_path.replace("/", "_")
        local_path = os.path.join(output_dir, local_rel)

        if os.path.exists(local_path):
            print(f"  [{i+1}/{total}] cached: {local_rel}")
        else:
            try:
                data = await storage_provider.load_file_to_memory(
                    folder=container, file_name=blob_path
                )
                with open(local_path, "wb") as f:
                    f.write(data)
                print(f"  [{i+1}/{total}] downloaded: {local_rel}")
            except Exception as e:
                print(f"  [{i+1}/{total}] FAILED: {local_rel} — {e}")
                local_path = None

        manifest.append({
            "local_path": local_path,
            "blob_url": blob_url,
            "query_index": rec["query_index"],
            "user_query": rec["user_query"],
            "image_agent_query": rec["image_agent_query"],
            "metadata": rec["metadata"],
        })

    return manifest


async def main():
    parser = argparse.ArgumentParser(description="Download vision test keyframes")
    parser.add_argument(
        "--log-dir",
        default="app/logs_GPT_4.1_no_critic_no_reflection_vision",
        help="Directory containing vision test event logs",
    )
    parser.add_argument(
        "--output-dir",
        default="keyframes_downloaded",
        help="Directory to save downloaded keyframe images",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output JSON manifest path (default: <output-dir>/manifest.json)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.output_dir, "manifest.json")

    # 1. Parse logs
    print(f"Parsing logs from {args.log_dir} ...")
    records = parse_logs(args.log_dir)
    print(f"Found {len(records)} unique keyframe images across {len(set(r['query_index'] for r in records))} queries\n")

    if not records:
        print("No keyframe images found in logs.")
        return

    # 2. Initialize storage provider
    settings = get_settings()
    credentials = resolve_credentials()
    storage_provider = AzureStorageProvider(
        storage_account_name=settings.storage_account_name,
        keyframe_container_name=settings.keyframe_container_name,
        credentials=credentials,
    )
    container_name = settings.keyframe_container_name

    # 3. Download
    print(f"Downloading to {args.output_dir}/ ...")
    manifest = await download_all(records, args.output_dir, storage_provider, container_name)

    # 4. Write manifest
    successful = [m for m in manifest if m["local_path"] is not None]
    failed = [m for m in manifest if m["local_path"] is None]

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {len(successful)} downloaded, {len(failed)} failed")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
