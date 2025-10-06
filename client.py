"""
Client for interacting with the MCP Server to get visual timestamps.

This client sends queries to the running MCP server and retrieves timestamps
of keyframes matching the query.

Usage:
    python client.py --query <query> [--top-k <number>] [--video-id <id>] [--youtube-url <url>]

Examples:
    python client.py --query 'person walking'
    python client.py --query 'car driving' --top-k 5
    python client.py --query 'person walking' --video-id video123
    python client.py --query 'sunset scene' --youtube-url 'https://youtube.com/watch?v=xyz'
"""

import httpx
import json
import sys
import argparse
from typing import Optional


class MCPClient:
    """Client for interacting with the MCP Server."""

    def __init__(self, server_url: str = "http://0.0.0.0:8000/mcp"):
        """
        Initialize the MCP client.

        Args:
            server_url: URL of the MCP server (default: http://0.0.0.0:8000/mcp)
        """
        self.server_url = server_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)
        self.session_id = None
        self._initialize_session()

    def _initialize_session(self):
        """Initialize MCP session and get session ID."""
        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "lively-client",
                    "version": "1.0.0"
                }
            }
        }

        try:
            response = self.client.post(
                self.server_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            response.raise_for_status()

            # Extract session ID from response header
            self.session_id = response.headers.get("mcp-session-id")
            if not self.session_id:
                raise Exception("Failed to get session ID from server")

            # Send initialized notification
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            self.client.post(
                self.server_url,
                json=notification,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": self.session_id
                }
            )

        except Exception as e:
            raise Exception(f"Failed to initialize MCP session: {str(e)}")

    def get_visual_timestamps(
        self,
        query: str,
        top_k: int = 3,
        video_id: Optional[str] = None,
        youtube_url: Optional[str] = None
    ) -> str:
        """
        Get timestamps for keyframes matching a text query.

        Args:
            query: Text query describing what to search for (e.g., "person walking", "car driving")
            top_k: Number of top results to return (default: 3)
            video_id: Optional video ID to filter results to a specific video
            youtube_url: Optional YouTube URL to filter results. Takes precedence over video_id if both provided.

        Returns:
            Comma-separated string of timestamps in seconds (e.g., "5.0, 12.5, 18.0")
        """
        # Prepare the MCP tool call request
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_visual_timestamps",
                "arguments": {
                    "query": query,
                    "top_k": top_k
                }
            }
        }

        # Add optional parameters if provided
        if youtube_url:
            payload["params"]["arguments"]["youtube_url"] = youtube_url
        elif video_id:
            payload["params"]["arguments"]["video_id"] = video_id

        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }

            # Include session ID if available
            if self.session_id:
                headers["mcp-session-id"] = self.session_id

            response = self.client.post(
                f"{self.server_url}",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            # Parse SSE response
            result = None
            for line in response.text.split('\n'):
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip():
                        result = json.loads(data)

            if not result:
                return "No response from server"

            # Extract the result from the MCP response
            if "error" in result:
                return f"Error: {result['error'].get('message', 'Unknown error')}"

            if "result" in result and "content" in result["result"]:
                # MCP returns content as a list of text items
                content = result["result"]["content"]
                if content and len(content) > 0:
                    return content[0].get("text", "No timestamps found")

            return "No timestamps found"

        except httpx.HTTPError as e:
            return f"HTTP Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Example usage of the MCP client."""
    parser = argparse.ArgumentParser(
        description="Search for video keyframes using natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --query 'person walking'
  python client.py --query 'car driving' --top-k 5
  python client.py --query 'person walking' --video-id video123
  python client.py --query 'sunset scene' --youtube-url 'https://youtube.com/watch?v=xyz'
        """
    )

    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Text query describing what to search for (e.g., "person walking", "car driving")'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top results to return (default: 3)'
    )
    parser.add_argument(
        '--video-id',
        type=str,
        default=None,
        help='Optional video ID to filter results to a specific video'
    )
    parser.add_argument(
        '--youtube-url',
        type=str,
        default=None,
        help='Optional YouTube URL to filter results. Takes precedence over video-id if both provided.'
    )
    parser.add_argument(
        '--server-url',
        type=str,
        default="http://0.0.0.0:8000/mcp",
        help='MCP server URL (default: http://0.0.0.0:8000/mcp)'
    )

    args = parser.parse_args()

    print(f"Searching for: '{args.query}'")
    print(f"Top K: {args.top_k}")
    if args.youtube_url:
        print(f"YouTube URL: {args.youtube_url}")
    elif args.video_id:
        print(f"Video ID: {args.video_id}")
    print("-" * 80)

    # Use the client
    with MCPClient(server_url=args.server_url) as client:
        timestamps = client.get_visual_timestamps(
            query=args.query,
            top_k=args.top_k,
            video_id=args.video_id,
            youtube_url=args.youtube_url
        )
        print(f"\nTimestamps: {timestamps}")

        # Parse timestamps if successful
        if timestamps and not timestamps.startswith("Error") and not timestamps.startswith("No"):
            timestamp_list = [t.strip() for t in timestamps.split(",")]
            print(f"Parsed list: {timestamp_list}")


if __name__ == "__main__":
    main()
