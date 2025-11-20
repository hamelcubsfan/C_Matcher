"""Application configuration helpers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


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

    database_url: str = ""  # This will be set by get_settings()

    upload_root: str = os.getenv("UPLOAD_ROOT", "uploads")

    app_shared_secret: str | None = os.getenv("APP_SHARED_SECRET")
    web_static_dir: str | None = os.getenv("WEB_STATIC_DIR", "apps/web/out")


import sys


def get_settings() -> Settings:
    """Creates a settings object and populates the database_url."""
    settings = Settings()

    if "DATABASE_URL" in os.environ:
        db_url = os.environ["DATABASE_URL"]
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        settings.database_url = db_url
        print(f"DATABASE_URL set from environment: {settings.database_url}", file=sys.stderr)
    elif "PGHOST" in os.environ:
        # Assumes standard PostgreSQL environment variables are set.
        user = os.environ.get("PGUSER", "postgres")
        password = os.environ.get("PGPASSWORD")
        host = os.environ["PGHOST"]
        port = os.environ.get("PGPORT", 5432)
        dbname = os.environ.get("PGDATABASE", "roles")

        if password:
            auth = f"{user}:{password}"
        else:
            auth = user
        settings.database_url = f"postgresql+psycopg://{auth}@{host}:{port}/{dbname}"
    else:
        # Default for local development
        print("WARNING: DATABASE_URL not found in environment. Available keys:", list(os.environ.keys()), file=sys.stderr)
        settings.database_url = "postgresql+psycopg://postgres:postgres@localhost:5432/roles"

    os.makedirs(settings.upload_root, exist_ok=True)
    return settings
