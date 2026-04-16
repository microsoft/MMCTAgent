"""TOON encoder - Token-Oriented Object Notation.

Implements encoding of Python data structures to TOON format,
a token-efficient alternative to JSON designed for LLM consumption.

TOON combines YAML-style indentation with CSV-style tabular arrays
to minimize token count while preserving data structure.

Spec: https://github.com/toon-format/spec
"""

from typing import Any, List, Dict, Set

# Characters that require quoting in TOON values
SPECIAL_CHARS: Set[str] = {',', ':', '{', '}', '[', ']', '\n', '\r', '\t', '"'}


def _needs_quoting(value: str) -> bool:
    """Check if string value needs quoting due to special characters."""
    if not value:
        return False
    return any(c in value for c in SPECIAL_CHARS)


def _quote(value: str) -> str:
    """Quote and escape a string value."""
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def _encode_scalar(value: Any) -> str:
    """Encode a scalar (primitive) value to TOON string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if value != value:  # NaN check
                return "null"
            if value == float('inf'):
                return "inf"
            if value == float('-inf'):
                return "-inf"
        return str(value)
    if isinstance(value, str):
        return _quote(value) if _needs_quoting(value) else value
    return _quote(str(value))


def _is_uniform_array(arr: List) -> bool:
    """Check if array contains uniform objects (all dicts with same keys)."""
    if not arr:
        return False
    if not isinstance(arr[0], dict):
        return False
    if len(arr) == 1:
        return True
    keys = set(arr[0].keys())
    return all(isinstance(item, dict) and set(item.keys()) == keys for item in arr[1:])


def _get_field_order(arr: List[Dict]) -> List[str]:
    """Get consistent field order for uniform array."""
    if not arr:
        return []
    all_keys = set(arr[0].keys())
    priority = ['id', 'node_id', 'video_id', 'score', 'title', 'name']
    ordered = [k for k in priority if k in all_keys]
    remaining = sorted(k for k in all_keys if k not in priority)
    return ordered + remaining


def _encode_uniform_array(key: str, arr: List[Dict], indent: int = 0) -> str:
    """Encode uniform array of objects as TOON table."""
    if not arr:
        return f"{'  ' * indent}{key}[0]{{}}:"
    fields = _get_field_order(arr)
    prefix = "  " * indent
    header = f"{prefix}{key}[{len(arr)}]{{{','.join(fields)}}}:"
    rows = []
    for item in arr:
        row_values = [_encode_scalar(item.get(f)) for f in fields]
        rows.append(f"{prefix}  {','.join(row_values)}")
    return header + "\n" + "\n".join(rows)


def _encode_simple_array(key: str, arr: List, indent: int = 0) -> str:
    """Encode array of scalars as compact TOON list."""
    prefix = "  " * indent
    encoded = [_encode_scalar(v) for v in arr]
    return f"{prefix}{key}[{len(arr)}]: {','.join(encoded)}"


def _encode_mixed_array(key: str, arr: List, indent: int = 0) -> str:
    """Encode non-uniform array with dash-prefixed items."""
    prefix = "  " * indent
    lines = [f"{prefix}{key}[{len(arr)}]:"]
    for item in arr:
        if isinstance(item, dict):
            if len(item) <= 3:
                pairs = [f"{k}: {_encode_scalar(v)}" for k, v in item.items()
                         if not isinstance(v, (dict, list))]
                if len(pairs) == len(item):
                    lines.append(f"{prefix}  - {{{', '.join(pairs)}}}")
                    continue
            lines.append(f"{prefix}  -")
            lines.append(_encode_object(item, indent + 2))
        elif isinstance(item, list):
            lines.append(f"{prefix}  - [{','.join(_encode_scalar(v) for v in item)}]")
        else:
            lines.append(f"{prefix}  - {_encode_scalar(item)}")
    return "\n".join(lines)


def _encode_object(obj: Dict, indent: int = 0) -> str:
    """Encode dictionary as YAML-style key: value pairs."""
    lines = []
    prefix = "  " * indent
    for key, value in obj.items():
        if isinstance(value, dict):
            if not value:
                lines.append(f"{prefix}{key}: {{}}")
            else:
                lines.append(f"{prefix}{key}:")
                lines.append(_encode_object(value, indent + 1))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{prefix}{key}[0]:")
            elif all(isinstance(v, dict) for v in value):
                if _is_uniform_array(value):
                    lines.append(_encode_uniform_array(key, value, indent))
                else:
                    lines.append(_encode_mixed_array(key, value, indent))
            elif all(not isinstance(v, (dict, list)) for v in value):
                lines.append(_encode_simple_array(key, value, indent))
            else:
                lines.append(_encode_mixed_array(key, value, indent))
        else:
            lines.append(f"{prefix}{key}: {_encode_scalar(value)}")
    return "\n".join(lines)


def to_toon(data: Any) -> str:
    """Convert Python object to TOON string.

    Args:
        data: Python object (dict, list, or scalar)

    Returns:
        TOON-encoded string
    """
    if isinstance(data, dict):
        return _encode_object(data)
    elif isinstance(data, list):
        if not data:
            return "items[0]:"
        if _is_uniform_array(data):
            return _encode_uniform_array("items", data)
        if all(not isinstance(v, (dict, list)) for v in data):
            return _encode_simple_array("items", data)
        return _encode_mixed_array("items", data)
    return _encode_scalar(data)
