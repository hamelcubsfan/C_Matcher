"""Application configuration helpers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit


@dataclass
class Settings:
    gh_board_url: str = os.getenv(
        "GH_BOARD_URL", "https://boards-api.greenhouse.io/v1/boards/waymo/jobs?content=true"
    )
    gh_harvest_api_key: str | None = os.getenv("GH_HARVEST_API_KEY")

    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    use_vertex_ai: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    google_project: str | None = os.getenv("GOOGLE_CLOUD_PROJECT")
    google_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))
    embed_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")

    rerank_provider: str = os.getenv("RERANK_PROVIDER", "local")

    database_url: str = ""  # populated by get_settings()

    upload_root: str = os.getenv("UPLOAD_ROOT", "/tmp" if os.getenv("VERCEL") else "uploads")

    app_shared_secret: str | None = os.getenv("APP_SHARED_SECRET")
    web_static_dir: str | None = os.getenv("WEB_STATIC_DIR", "apps/web/out")


def _redact_db_url(url: str) -> str:
    """Remove password from DATABASE_URL before printing to logs."""
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return "<set>"

        username = parts.username or ""
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        auth = f"{username}:***@" if username else ""
        netloc = f"{auth}{host}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<set>"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create and cache settings, including database_url."""
    settings = Settings()

    if "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        settings.database_url = db_url
        print(
            f"DATABASE_URL set from environment: {_redact_db_url(settings.database_url)}",
            file=sys.stderr,
        )
    elif "PGHOST" in os.environ:
        user = os.environ.get("PGUSER", "postgres")
        password = os.environ.get("PGPASSWORD")
        host = os.environ["PGHOST"]
        port = os.environ.get("PGPORT", 5432)
        dbname = os.environ.get("PGDATABASE", "roles")

        auth = f"{user}:{password}" if password else user
        settings.database_url = f"postgresql+psycopg2://{auth}@{host}:{port}/{dbname}"
    else:
        print("DATABASE_URL not found in environment. Using local fallback.", file=sys.stderr)
        settings.database_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/roles"

    try:
        os.makedirs(settings.upload_root, exist_ok=True)
    except OSError as e:
        if e.errno == 30:
            print(f"{settings.upload_root} is read-only. Falling back to /tmp", file=sys.stderr)
            settings.upload_root = "/tmp"
            os.makedirs(settings.upload_root, exist_ok=True)
        else:
            raise

    return settings
