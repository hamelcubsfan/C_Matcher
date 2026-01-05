"""Vercel entrypoint for the FastAPI application."""
from services.api.main import app

# Vercel rewrites /api/* to this function, so we need to tell FastAPI about the prefix
app.root_path = "/api"

from services.shared.db import get_engine, Base
from services.shared.models import Candidate, Job  # Import models to register them with Base

@app.on_event("startup")
def init_db():
    # Create tables if they don't exist
    Base.metadata.create_all(bind=get_engine())
