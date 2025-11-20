"""Local file storage utilities for resumes."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from .config import get_settings


settings = get_settings()


def save_upload(file_obj, filename: str | None = None) -> str:
    """Persist an uploaded file and return the relative path."""
    upload_root = Path(settings.upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    suffix = Path(file_obj.filename or "upload").suffix
    name = filename or f"{uuid.uuid4()}{suffix}"
    target = upload_root / name

    with target.open("wb") as buffer:
        shutil.copyfileobj(file_obj.file, buffer)

    return str(target)


def load_text_from_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to parse resumes") from exc

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
