"""Vercel entrypoint for the FastAPI application."""
from services.api.main import app
import sys
print("DEBUG: Loading api/main.py (Vercel Entrypoint)", file=sys.stderr)

# Vercel rewrites /api/* to this function, so we need to tell FastAPI about the prefix
app.root_path = "/api"


