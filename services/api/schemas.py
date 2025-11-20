"""API-facing Pydantic models."""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from services.shared.schemas import CandidateRead, MatchResult


class UploadResponse(BaseModel):
    candidate: CandidateRead


class MatchResponse(BaseModel):
    candidate_id: uuid.UUID
    results: list[MatchResult]


class RouteResponse(BaseModel):
    routed: bool
    note: str
    job_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
