"""Utilities for saving and cleaning up uploaded files on the local filesystem."""

import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile

UPLOAD_DIR = Path(os.getenv("MMCT_UPLOAD_DIR", "uploads"))


async def save_upload_file(upload_file: UploadFile, subdir: str = "") -> Path:
    """
    Persist an UploadFile to UPLOAD_DIR/<subdir>/<uuid>_<original_filename>.

    Returns the absolute Path to the saved file.
    Raises HTTP 400 if the file content is empty.
    """
    dest_dir = UPLOAD_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{Path(upload_file.filename or 'upload').name}"
    dest_path = dest_dir / safe_name

    contents = await upload_file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(contents)

    return dest_path.resolve()


async def delete_file(path: Path) -> None:
    """Remove a file silently; does not raise on failure."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
