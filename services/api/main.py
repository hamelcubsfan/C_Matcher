"""FastAPI gateway for the Waymo Role Matcher."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import docx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.api.dependencies import get_db_session
from services.api.schemas import RouteResponse, UploadResponse
from services.ingest.run_once import run_once as ingest_run_once
from services.match.service import match_candidate
from services.shared.config import get_settings
from services.shared.db import get_session_factory, reset_db
from services.shared.embeddings import embed_texts
from services.shared.models import Candidate, Job
from services.shared.resume import extract_candidate_info, extract_skills, summarize_text
from services.shared.schemas import CandidateRead
from services.shared.storage import load_text_from_pdf, save_upload

app = FastAPI(title="Waymo Role Matcher API")
settings = get_settings()
STATIC_WEB_DIR = Path(settings.web_static_dir).resolve() if settings.web_static_dir else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLEAR_CANDIDATES_ON_STARTUP = os.getenv("CLEAR_CANDIDATES_ON_STARTUP", "false").lower() == "true"


@app.on_event("startup")
async def start_background_ingest():
    # Optional candidate cleanup (off by default for serverless cold starts)
    if CLEAR_CANDIDATES_ON_STARTUP:
        try:
            with get_session_factory()() as session:
                session.query(Candidate).delete()
                session.commit()
        except Exception as e:
            print(f"Could not clear candidates: {e}", file=sys.stderr)

    asyncio.create_task(background_ingest_loop())


async def background_ingest_loop():
    while True:
        try:
            print("Starting background ingestion...", file=sys.stderr)
            count = await asyncio.to_thread(ingest_run_once, settings.gh_board_url)
            print(f"Background ingestion complete. Ingested {count} jobs.", file=sys.stderr)
        except Exception as e:
            print(f"Error in background ingestion: {e}", file=sys.stderr)

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
    info = extract_candidate_info(text)
    skills = extract_skills(text)

    semantic_summary = f"Candidate Name: {info.name}\n"

    if info.experience:
        current_role = info.experience[0]
        semantic_summary += f"Current Role: {current_role.title}\n"

    if info.summary:
        semantic_summary += f"Professional Summary: {info.summary}\n"

    if notes:
        semantic_summary += f"\nRECRUITER NOTES (IMPORTANT CONTEXT): {notes}\n"

    semantic_summary += "Work Experience:\n"
    for exp in info.experience:
        semantic_summary += f"- Role: {exp.title} at {exp.company}\n"
        if exp.description:
            semantic_summary += f"  Details: {exp.description[:200]}...\n"

    semantic_summary += f"\nSkills: {', '.join(skills[:10])}"

    if len(semantic_summary) < 100:
        semantic_summary = summarize_text(text)

    embedding = embed_texts([semantic_summary])[0]

    profile_data = info.model_dump()
    profile_data["raw_text"] = text
    if notes:
        profile_data["recruiter_notes"] = notes

    # Single active candidate policy (kept here, not on startup)
    session.query(Candidate).delete()

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


@app.post("/match/{candidate_id}")
def match_candidate_endpoint(candidate_id: uuid.UUID, limit: int = 5):
    """
    Match a candidate against jobs and stream progress via SSE.

    Important: do NOT use a request-scoped DB dependency for streaming.
    Keep a session open inside the generator for the lifetime of the stream.
    """
    factory = get_session_factory()

    # Preflight check to return a clean 404 before starting the stream
    pre = factory()
    try:
        candidate = pre.get(Candidate, candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
    finally:
        pre.close()

    def event_generator():
        session = factory()
        try:
            for event in match_candidate(session, candidate_id, limit):
                yield f"data: {json.dumps(event)}\n\n"
            session.commit()
        except Exception as e:
            session.rollback()
            yield f"data: {json.dumps({'status': 'error', 'detail': str(e)})}\n\n"
        finally:
            session.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
