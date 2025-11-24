"""Utilities for fetching and normalizing Greenhouse jobs."""
from __future__ import annotations

from typing import Iterable

import sys
import requests

from services.shared.config import get_settings
from services.shared.embeddings import embed_texts


def fetch_jobs(board_url: str | None = None) -> list[dict]:
    url = board_url or get_settings().gh_board_url
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    all_jobs = payload.get("jobs", [])
    
    # Filter out internships
    filtered_jobs = []
    print(f"Fetching jobs from {url}...", file=sys.stderr)
    for job in all_jobs:
        title = job.get("title", "").lower()
        if "intern" in title:
            print(f"Filtering out intern job: {title}", file=sys.stderr)
            continue
        filtered_jobs.append(job)
    print(f"Kept {len(filtered_jobs)} jobs out of {len(all_jobs)}.", file=sys.stderr)
        
    return filtered_jobs


def normalize(job: dict) -> dict:
    departments = job.get("departments") or []
    team = departments[0].get("name") if departments else None
    location = job.get("location", {}).get("name") if job.get("location") else None

    metadata = job.get("metadata") or []
    must_have: list[str] = []
    nice_to_have: list[str] = []
    for item in metadata:
        name = (item.get("name") or "").lower()
        value = item.get("value")
        if not value:
            continue
        if "must" in name:
            must_have.append(value)
        elif "nice" in name or "preferred" in name:
            nice_to_have.append(value)

    return {
        "greenhouse_job_id": str(job.get("id")),
        "requisition_id": job.get("requisition_id"),
        "title": job.get("title", ""),
        "team": team,
        "location": location,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
        "description": job.get("content", ""),
        "absolute_url": job.get("absolute_url"),
    }


def embed_documents(jobs: Iterable[dict]) -> list[list[float]]:
    docs = [f"{job['title']}\n{job.get('description', '')}" for job in jobs]
    return embed_texts(docs)
