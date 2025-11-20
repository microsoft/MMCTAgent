"""Helpers for loading lightweight transcript artifacts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptDocument:
    segments: List[TranscriptSegment]

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        return max(segment.end for segment in self.segments)

    def chunk_by_duration(self, max_duration: float) -> Iterable[List[TranscriptSegment]]:
        chunk: List[TranscriptSegment] = []
        window_start = None
        for segment in self.segments:
            if not chunk:
                window_start = segment.start
            chunk.append(segment)
            if window_start is not None and segment.end - window_start >= max_duration:
                yield chunk
                chunk = []
                window_start = None
        if chunk:
            yield chunk


_SRT_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)


def _parse_srt_timestamp(raw: str) -> float:
    match = _SRT_TIME_RE.match(raw.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {raw}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _parse_srt(path: Path) -> List[TranscriptSegment]:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: List[TranscriptSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing_line = None
        text_lines = []
        # Skip numeric line numbers if present
        if re.match(r"^\d+$", lines[0]):
            if len(lines) >= 2:
                timing_line = lines[1]
                text_lines = lines[2:]
            else:
                continue
        else:
            timing_line = lines[0]
            text_lines = lines[1:]
        if "-->" not in timing_line:
            continue
        start_raw, end_raw = [part.strip() for part in timing_line.split("-->")]
        start = _parse_srt_timestamp(start_raw)
        end = _parse_srt_timestamp(end_raw)
        text = " ".join(text_lines).strip()
        if not text:
            continue
        segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def _parse_json_transcript(path: Path) -> List[TranscriptSegment]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        TranscriptSegment(
            start=float(entry.get("start", 0.0)),
            end=float(entry.get("end", entry.get("start", 0.0))),
            text=str(entry.get("text", "")).strip(),
        )
        for entry in payload.get("segments", [])
    ]


def load_transcript(path: str | Path) -> TranscriptDocument:
    """Load a transcript from JSON or SRT sources."""

    doc_path = Path(path)
    suffix = doc_path.suffix.lower()
    if suffix == ".srt":
        segments = _parse_srt(doc_path)
    elif suffix == ".json":
        segments = _parse_json_transcript(doc_path)
    else:
        raise ValueError(
            f"Unsupported transcript format '{suffix}'. Only .json and .srt are supported."
        )

    return TranscriptDocument(segments=segments)
