"""Vercel entrypoint for the FastAPI application."""
from services.api.main import app
import sys
print("DEBUG: Loading api/main.py (Vercel Entrypoint)", file=sys.stderr)

# Vercel rewrites /api/* to this function, so we need to tell FastAPI about the prefix
app.root_path = "/api"

from services.shared.db import get_engine, Base
from services.shared.models import Candidate, Job

@app.on_event("startup")
def init_db():
    try:
        from sqlalchemy import text
        with get_engine().connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
        
        Base.metadata.create_all(bind=get_engine())
    except Exception as e:
        import sys
        print(f"DB Init Error: {e}", file=sys.stderr)


