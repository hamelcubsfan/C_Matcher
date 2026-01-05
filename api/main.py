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
    import sys
    print("DEBUG: Running init_db in api/main.py", file=sys.stderr)
    try:
        from sqlalchemy import text
        with get_engine().connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.commit()
            print("DEBUG: 'vector' extension enabled.", file=sys.stderr)
        
        Base.metadata.create_all(bind=get_engine())
        print("DEBUG: Tables created successfully", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Table creation failed: {e}", file=sys.stderr)


