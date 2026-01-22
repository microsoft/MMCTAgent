# V2 API Reference

This document describes the V2 multi-agent Video and Image QA API endpoints for frontend integration.

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### 1. Non-Streaming Query

**POST** `/v2/query`

Unified endpoint for video and image queries using the V2 multi-agent system. Returns a complete response after all agents finish processing.

#### Request

**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Natural language query about the video/image |
| `video_id` | string | ❌ | null | Unique identifier for an indexed video |
| `url` | string | ❌ | null | Video URL (YouTube or other source) |
| `use_critic_agent` | boolean | ❌ | true | Enable critic agent for response validation |
| `cache` | boolean | ❌ | false | Enable caching for repeated queries |
| `file` | file | ❌ | null | Optional image file to query |

#### Example Request (cURL)

```bash
# Query with video_id
curl -X POST "http://localhost:8000/v2/query" \
  -F "query=What is the main topic of this video?" \
  -F "video_id=abc123" \
  -F "use_critic_agent=true"

# Query with image file
curl -X POST "http://localhost:8000/v2/query" \
  -F "query=What objects are in this image?" \
  -F "file=@/path/to/image.jpg" \
  -F "use_critic_agent=true"
```

#### Example Request (JavaScript fetch)

```javascript
const formData = new FormData();
formData.append('query', 'What is the main topic of this video?');
formData.append('video_id', 'abc123');
formData.append('use_critic_agent', 'true');

const response = await fetch('/v2/query', {
  method: 'POST',
  body: formData
});

const data = await response.json();
```

#### Response

**Content-Type:** `application/json`

```json
{
  "response": "The tradition of decorating Christmas trees originated from \"paradise trees\" used in church plays [1]. These trees were adorned with candles, apples, and berries, later adopted into homes [1].",
  "answer_found": true,
  "sources": [
    {
      "citation_id": 1,
      "video_id": "808ef24205b8bfe7181818699675f5a4dbfe5974baf5ded99ab5b5b3c8b6f15d",
      "url": "https://www.youtube.com/watch?v=fFjv93ACGo8",
      "start_time": "00:00:32",
      "end_time": "00:01:13"
    }
  ],
  "token_usage": {
    "prompt_tokens": 24235,
    "completion_tokens": 1161
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Markdown-formatted answer with inline citations `[1]`, `[2]`, etc. |
| `answer_found` | boolean | Whether the query was answered from available context |
| `sources` | array | List of citation sources with timestamps |
| `sources[].citation_id` | integer | Citation number matching `[n]` in response |
| `sources[].video_id` | string | Hash identifier of the video |
| `sources[].url` | string | Original video URL (if available) |
| `sources[].start_time` | string | Start timestamp in `HH:MM:SS` format |
| `sources[].end_time` | string | End timestamp in `HH:MM:SS` format |
| `token_usage` | object | Total token consumption stats |
| `token_usage.prompt_tokens` | integer | Total input tokens used |
| `token_usage.completion_tokens` | integer | Total output tokens generated |

---

### 2. Streaming Query (SSE)

**POST** `/v2/query/stream`

Streaming endpoint that sends real-time agent logs as Server-Sent Events (SSE). Ideal for showing live agent activity in the UI.

#### Request

**Content-Type:** `multipart/form-data`

Same parameters as the non-streaming endpoint.

#### Example Request (cURL)

```bash
curl -X POST "http://localhost:8000/v2/query/stream" \
  -F "query=What is the history of Christmas trees?" \
  -F "video_id=abc123" \
  -F "use_critic_agent=true" \
  -H "Accept: text/event-stream"
```

#### Example Request (JavaScript EventSource)

```javascript
// Using fetch with ReadableStream for POST requests
async function streamQuery(query, videoId) {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('video_id', videoId);
  formData.append('use_critic_agent', 'true');

  const response = await fetch('/v2/query/stream', {
    method: 'POST',
    body: formData
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // Keep incomplete event in buffer

    for (const line of lines) {
      if (!line.trim()) continue;
      
      const eventMatch = line.match(/event: (\w+)/);
      const dataMatch = line.match(/data: (.+)/);
      
      if (eventMatch && dataMatch) {
        const eventType = eventMatch[1];
        const data = JSON.parse(dataMatch[1]);
        
        handleEvent(eventType, data);
      }
    }
  }
}

function handleEvent(eventType, data) {
  switch (eventType) {
    case 'connected':
      console.log('Stream connected:', data.message);
      break;
    case 'message':
      console.log(`[${data.source}]: ${data.content}`);
      break;
    case 'tool_call':
      console.log(`[${data.source}] calling tools:`, data.tool_names);
      break;
    case 'tool_result':
      console.log(`Tool results:`, data.results);
      break;
    case 'handoff':
      console.log(`[${data.source}] → ${data.target}`);
      break;
    case 'result':
      console.log('Final result:', data.content);
      console.log('Token usage:', data.token_usage);
      break;
    case 'complete':
      console.log('Stream complete');
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
}
```

#### Response

**Content-Type:** `text/event-stream`

The stream emits events in SSE format:

```
event: <event_type>
data: <json_payload>

```

#### Event Types

##### `connected`
Initial connection confirmation.

```
event: connected
data: {"message": "Stream connected", "query": "What is...", "timestamp": "2026-01-22T17:00:00.000000"}
```

##### `message`
Text message from an agent.

```
event: message
data: {"type": "message", "source": "planner", "content": "I'll start by querying the VideoAgent...", "timestamp": "..."}
```

##### `tool_call`
Agent requesting tool execution.

```
event: tool_call
data: {"type": "tool_call", "source": "VideoAgent", "tool_names": ["search_video_context"], "tools": [{"name": "search_video_context", "arguments": "{\"query\": \"...\"}"}], "timestamp": "..."}
```

##### `tool_result`
Results from tool execution.

```
event: tool_result
data: {"type": "tool_result", "source": "VideoAgent", "results": [{"call_id": "...", "content": "..."}], "timestamp": "..."}
```

##### `handoff`
Agent handing off to another agent.

```
event: handoff
data: {"type": "handoff", "source": "planner", "target": "VideoAgent", "content": "Transferred to VideoAgent", "timestamp": "..."}
```

##### `result`
Final result containing the answer and token usage.

```
event: result
data: {
  "type": "result",
  "source": "system",
  "content": {
    "response": "The tradition of decorating Christmas trees...",
    "answer_found": true,
    "sources": [...]
  },
  "message_count": 22,
  "stop_reason": "Text 'TERMINATE' mentioned",
  "duration_seconds": 41.95,
  "token_usage": {
    "prompt_tokens": 21754,
    "completion_tokens": 1124
  },
  "timestamp": "..."
}
```

##### `complete`
Stream completion marker.

```
event: complete
data: {"message": "Query processing complete", "timestamp": "..."}
```

##### `error`
Error occurred during processing.

```
event: error
data: {"message": "Error description", "timestamp": "..."}
```

---

## React Integration Example

```tsx
import { useState, useCallback } from 'react';

interface Citation {
  citation_id: number;
  video_id: string;
  url: string;
  start_time: string;
  end_time: string;
}

interface AgentLog {
  type: string;
  source: string;
  content?: string;
  target?: string;
  tool_names?: string[];
  timestamp: string;
}

interface QueryResult {
  response: string;
  answer_found: boolean;
  sources: Citation[];
  token_usage: {
    prompt_tokens: number;
    completion_tokens: number;
  };
}

export function useV2Query() {
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const streamQuery = useCallback(async (query: string, videoId?: string) => {
    setIsLoading(true);
    setLogs([]);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('query', query);
    if (videoId) formData.append('video_id', videoId);
    formData.append('use_critic_agent', 'true');

    try {
      const response = await fetch('/v2/query/stream', {
        method: 'POST',
        body: formData
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop()!;

        for (const event of events) {
          if (!event.trim()) continue;

          const eventMatch = event.match(/event: (\w+)/);
          const dataMatch = event.match(/data: (.+)/s);

          if (eventMatch && dataMatch) {
            const eventType = eventMatch[1];
            const data = JSON.parse(dataMatch[1]);

            if (eventType === 'result') {
              setResult({
                ...data.content,
                token_usage: data.token_usage
              });
            } else if (eventType === 'error') {
              setError(data.message);
            } else if (['message', 'tool_call', 'handoff', 'tool_result'].includes(eventType)) {
              setLogs(prev => [...prev, data]);
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { streamQuery, logs, result, isLoading, error };
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Notes

1. **Citations**: The `response` field contains inline citations like `[1]`, `[2]` that map to entries in the `sources` array by `citation_id`.

2. **Token Usage**: Both endpoints return `token_usage` with total prompt and completion tokens consumed across all agents.

3. **Streaming Performance**: The streaming endpoint allows showing real-time agent activity, improving perceived responsiveness for long-running queries.

4. **Image Queries**: Upload an image file using the `file` parameter for image-based queries. The system will use the ImageAgent for visual analysis.

5. **Video Queries**: Provide either `video_id` (for pre-indexed videos) or `url` (for new videos) to query video content.
