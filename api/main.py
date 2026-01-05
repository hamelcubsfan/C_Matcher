"""Vercel entrypoint for the FastAPI application."""
from services.api.main import app

# Vercel rewrites /api/* to this function, so we need to tell FastAPI about the prefix
app.root_path = "/api"
