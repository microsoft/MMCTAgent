"""Internal formatting helpers shared between orchestrator and agents."""

import re
from collections import defaultdict
from typing import Any, Dict, List


def format_evidence_compact(evidence: List[Dict[str, Any]]) -> str:
    """Format evidence list as compact text instead of verbose JSON.

    For Chapter nodes with timestamped descriptions ([Xs] markers),
    chapter-level start/end times are omitted so the LLM must use
    the specific [Xs] timestamps for citations.
    """
    if not evidence:
        return "No evidence retrieved."

    lines = []
    for item in evidence:
        node_type = item.get("node_type", "?")
        video_id = item.get("video_id", "?")
        score = item.get("score")
        start = item.get("start_time")
        end = item.get("end_time")
        chunk_idx = item.get("chunk_index")
        node_id = item.get("node_id", "")

        content = (
            item.get("summary")
            or item.get("transcript")
            or item.get("description")
            or ""
        )

        parts = [node_type, f"video:{video_id}"]
        if chunk_idx is not None:
            parts.append(f"chunk:{chunk_idx}")
        has_inline_timestamps = "[" in content and "s]" in content
        if start is not None and end is not None and not has_inline_timestamps:
            parts.append(f"{start}-{end}s")
        if score is not None:
            parts.append(f"score:{score:.2f}")
        parts.append(f"id:{node_id}")
        header = " | ".join(parts)

        lines.append(f"[{header}]")
        if content:
            lines.append(content)
        scene = item.get("scene_composition")
        if scene:
            lines.append(f"Scene: {scene}")
        ocr = item.get("ocr_data")
        if ocr:
            lines.append(f"Visible text: {ocr}")
        lines.append("")

    return "\n".join(lines).rstrip()


def dedup_sources(answer: str, sources: List[Dict]) -> tuple[str, List[Dict]]:
    """Merge all citations for the same video into a single citation.

    For each video_id, produces one citation whose start_time is the
    minimum across all original citations and end_time is the maximum.
    The answer text is updated so all old markers map to the new numbers.
    """
    if len(sources) <= 1:
        return answer, sources

    seen_order: List[str] = []
    groups: Dict[str, List[tuple[int, Dict]]] = defaultdict(list)
    for idx, src in enumerate(sources):
        vid = src["video_id"]
        if vid not in groups:
            seen_order.append(vid)
        groups[vid].append((idx, src))

    old_to_new: Dict[int, int] = {}
    merged: List[Dict] = []

    for vid in seen_order:
        items = groups[vid]
        new_num = len(merged) + 1
        start = min(s.get("start_time", 0) for _, s in items)
        end = max(s.get("end_time", 0) for _, s in items)
        for old_idx, _ in items:
            old_to_new[old_idx + 1] = new_num
        merged.append({
            "citation": f"[{new_num}]",
            "video_id": vid,
            "start_time": start,
            "end_time": end,
        })

    valid_nums = set(range(1, len(merged) + 1))

    def _replace_marker(m: re.Match) -> str:
        n = int(m.group(1))
        mapped = old_to_new.get(n)
        if mapped is not None:
            return f"[{mapped}]"
        if n in valid_nums:
            return f"[{n}]"
        return ""

    new_answer = re.sub(r"\[(\d+)\]", _replace_marker, answer)
    new_answer = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", new_answer)
    new_answer = re.sub(r"  +", " ", new_answer)

    return new_answer, merged
