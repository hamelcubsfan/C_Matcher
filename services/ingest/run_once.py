"""One-shot ingestion runner."""
from __future__ import annotations

from sqlalchemy import select

from services.shared import db
from services.shared.config import get_settings
from services.shared.models import Job

from .fetch_jobs import embed_documents, fetch_jobs, normalize


def upsert_jobs(session, jobs: list[dict], embeddings: list[list[float]]) -> None:
    for job_dict, embedding in zip(jobs, embeddings, strict=False):
        stmt = select(Job).where(Job.greenhouse_job_id == job_dict["greenhouse_job_id"])
        existing = session.execute(stmt).scalar_one_or_none()
        if existing:
            for key, value in job_dict.items():
                setattr(existing, key, value)
            existing.embedding = embedding
        else:
            session.add(Job(**job_dict, embedding=embedding))


def run_once(board_url: str | None = None) -> int:
    settings = get_settings()
    with db.session_scope() as session:
        raw_jobs = fetch_jobs(board_url or settings.gh_board_url)
        normalized = [normalize(job) for job in raw_jobs]
        embeddings = embed_documents(normalized)
        upsert_jobs(session, normalized, embeddings)
        return len(normalized)


if __name__ == "__main__":
    count = run_once()
    print(f"Ingested {count} jobs")
