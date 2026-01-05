"""FastAPI gateway for the Waymo Role Matcher."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel

from services.api.dependencies import get_db_session
from services.api.schemas import MatchResponse, RouteResponse, UploadResponse
from services.ingest.run_once import run_once as ingest_run_once
from services.match.service import match_candidate
from services.shared.config import get_settings
from services.shared.db import get_session_factory, reset_db, get_engine, Base

from services.shared.embeddings import embed_texts
from services.shared.models import Candidate, Job
from services.shared.resume import extract_skills, summarize_text, extract_candidate_info
import docx

from services.shared.schemas import CandidateRead
from services.shared.storage import load_text_from_pdf, save_upload

app = FastAPI(title="Waymo Role Matcher API")
settings = get_settings()
STATIC_WEB_DIR = (
    Path(settings.web_static_dir).resolve() if settings.web_static_dir else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import asyncio

@app.on_event("startup")
async def start_background_ingest():
    # We moved DB init to api/main.py to avoid stale code issues
    
    # Clear candidates on startup for privacy/hygiene
    # Wrap in try/except to prevent crash if tables don't exist yet (though they should)
    try:
        with get_session_factory()() as session:
            session.query(Candidate).delete()
            session.commit()
    except Exception as e:
        print(f"WARNING: Could not clear candidates: {e}", file=sys.stderr)

    asyncio.create_task(background_ingest_loop())

async def background_ingest_loop():
    while True:
        try:
            print("Starting background ingestion...", file=sys.stderr)
            # ingest_run_once is synchronous, so run it in a thread to not block the event loop
            count = await asyncio.to_thread(ingest_run_once, settings.gh_board_url)
            print(f"Background ingestion complete. Ingested {count} jobs.", file=sys.stderr)
        except Exception as e:
            print(f"Error in background ingestion: {e}", file=sys.stderr)
        
        # Sleep for 24 hours
        await asyncio.sleep(86400)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _service_metadata() -> dict:
    return {
        "service": "Waymo Role Matcher API",
        "docs_url": "/docs",
        "health_url": "/health",
        "message": "See /docs for interactive API explorer.",
    }


@app.get("/api-info")
def api_info() -> dict:
    return _service_metadata()


@app.post("/ingest/run")
def trigger_ingest() -> dict:
    count = ingest_run_once(settings.gh_board_url)
    return {"ingested": count}


@app.post("/admin/reset-db")
def reset_database(secret: str) -> dict:
    """Reset the database schema."""
    if not settings.app_shared_secret or secret != settings.app_shared_secret:
        raise HTTPException(status_code=403, detail="Invalid secret")
    reset_db(get_session_factory())
    return {"status": "ok", "message": "Database re-initialized."}


def _load_resume_text(path: str, content_type: str | None) -> str:
    if content_type and "pdf" in content_type.lower():
        return load_text_from_pdf(path)
    lower_path = path.lower()
    if lower_path.endswith(".pdf"):
        return load_text_from_pdf(path)
    if lower_path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


        return UploadResponse(candidate=CandidateRead.model_validate(candidate))
    finally:
        file.file.close()


class PasteRequest(BaseModel):
    text: str
    notes: str | None = None


@app.post("/candidates/paste", response_model=UploadResponse)
def paste_candidate(
    request: PasteRequest,
    session: Session = Depends(get_db_session),
) -> UploadResponse:
    """Process a pasted resume text."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty")
    
    return _process_candidate_text(request.text, "pasted_text", session, notes=request.notes)


def _process_candidate_text(text: str, source: str, session: Session, notes: str | None = None) -> UploadResponse:
    """Shared logic to process candidate text from file or paste."""
    # Extract info and skills
    info = extract_candidate_info(text)
    skills = extract_skills(text)

    # Construct a semantic summary for embedding that emphasizes the ROLE over raw keywords
    # This helps prevent Recruiters with "AI" experience from being matched to "AI Engineer" roles
    semantic_summary = f"Candidate Name: {info.name}\n"
    
    # Boost the most recent role by putting it first and explicitly labeling it
    if info.experience:
        current_role = info.experience[0]
        semantic_summary += f"Current Role: {current_role.title}\n"

    if info.summary:
        semantic_summary += f"Professional Summary: {info.summary}\n"
    
    # Inject Recruiter Notes if present - HIGH PRIORITY
    if notes:
        semantic_summary += f"\nRECRUITER NOTES (IMPORTANT CONTEXT): {notes}\n"

    semantic_summary += "Work Experience:\n"
    for exp in info.experience:
        semantic_summary += f"- Role: {exp.title} at {exp.company}\n"
        if exp.description:
            semantic_summary += f"  Details: {exp.description[:200]}...\n" # Truncate description to focus on role
    
    # Reduce skills noise (limit to 10 instead of 20) to prevent keyword flooding
    semantic_summary += f"\nSkills: {', '.join(skills[:10])}"
    
    # Fallback if extraction failed
    if len(semantic_summary) < 100:
        semantic_summary = summarize_text(text)

    embedding = embed_texts([semantic_summary])[0]
    
    profile_data = info.model_dump()
    profile_data["raw_text"] = text  # Restore raw text for reranker
    if notes:
        profile_data["recruiter_notes"] = notes # Store notes in profile for display/debugging

    # Enforce Single Active Candidate Policy
    # Delete ALL existing candidates to ensure we only have the current one
    session.query(Candidate).delete()
    
    # Create new candidate
    candidate = Candidate(
        resume_url=source,
        parsed_profile=profile_data,
        full_name=info.name,
        email=info.email,
        skills=skills,
        embedding=embedding,
    )
    session.add(candidate)
        
    session.flush()
    session.refresh(candidate)
    return UploadResponse(candidate=CandidateRead.model_validate(candidate))


@app.post("/candidates/upload", response_model=UploadResponse)
def upload_candidate(
    file: UploadFile = File(...),
    notes: str | None = Form(None),
    session: Session = Depends(get_db_session),
) -> UploadResponse:
    """Upload a resume, parse it, and store a candidate record."""
    try:
        stored_path = save_upload(file)
        text = _load_resume_text(stored_path, file.content_type)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Unable to extract text from resume")

        return _process_candidate_text(text, stored_path, session, notes=notes)
    finally:
        file.file.close()


@app.get("/candidates/{candidate_id}", response_model=CandidateRead)
def get_candidate(candidate_id: uuid.UUID, session: Session = Depends(get_db_session)) -> CandidateRead:
    candidate = session.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateRead.model_validate(candidate)


from fastapi.responses import StreamingResponse
import json
import logging
logger = logging.getLogger(__name__)

@app.post("/match/{candidate_id}")
def match_candidate_endpoint(candidate_id: uuid.UUID, limit: int = 5, session: Session = Depends(get_db_session)):
    """Match a candidate against open jobs."""
    try:
        def event_generator():
            for event in match_candidate(session, candidate_id, limit):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Match failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route/{candidate_id}/{job_id}", response_model=RouteResponse)
def route_candidate(
    candidate_id: uuid.UUID,
    job_id: int,
    session: Session = Depends(get_db_session),
) -> RouteResponse:
    candidate = session.get(Candidate, candidate_id)
    job = session.get(Job, job_id)
    if not candidate or not job:
        raise HTTPException(status_code=404, detail="Candidate or job not found")

    note = json.dumps(
        {
            "candidate": str(candidate_id),
            "job": job.title,
            "job_url": job.absolute_url,
        }
    )
    return RouteResponse(routed=False, note=note, job_url=job.absolute_url)


if STATIC_WEB_DIR and STATIC_WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_WEB_DIR, html=True), name="web")
else:

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return _service_metadata()
