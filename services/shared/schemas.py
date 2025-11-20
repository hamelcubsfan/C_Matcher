"""Pydantic schemas shared across services."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JobRead(BaseModel):
    id: int
    greenhouse_job_id: str
    title: str
    team: str | None
    location: str | None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    description: str | None
    absolute_url: str | None
    posting_status: str

    class Config:
        from_attributes = True


class CandidateRead(BaseModel):
    id: uuid.UUID
    full_name: str | None
    email: str | None
    location: str | None
    resume_url: str | None
    parsed_profile: dict[str, Any] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MatchResult(BaseModel):
    job: JobRead
    retrieval_score: float | None = None
    rerank_score: float | None = None
    confidence: float | None = None
    explanation: str | None = None
    reason_codes: list[dict[str, Any]] = Field(default_factory=list)
