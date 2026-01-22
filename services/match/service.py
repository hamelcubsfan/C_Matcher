"""Candidate/job matching pipeline."""
from __future__ import annotations

import os
import time
import uuid
from typing import Iterator

import concurrent.futures
from concurrent.futures import FIRST_COMPLETED, wait

from sqlalchemy import cast, func, select
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from services.explain.service import ExplanationService
from services.match.rerank import get_reranker
from services.shared.config import get_settings
from services.shared.embeddings import embed_texts
from services.shared.models import Candidate, Job, Match
from services.shared.resume import chunk_with_labels, summarize_text
from services.shared.schemas import MatchResult

_settings = get_settings()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


MAX_JOBS_TO_CHECK = _env_int("MAX_JOBS_TO_CHECK", 25)
EXPLAIN_WORKERS = _env_int("EXPLAIN_WORKERS", 4)
EXPLAIN_STAGE_TIMEOUT_S = _env_float("EXPLAIN_STAGE_TIMEOUT_S", 120.0)
MATCH_TOTAL_BUDGET_S = _env_float("MATCH_TOTAL_BUDGET_S", 240.0)
MIN_LLM_CONFIDENCE = _env_float("MIN_LLM_CONFIDENCE", 0.7)


class MatchService:
    def __init__(self, session: Session):
        self.session = session
        self.reranker = get_reranker()
        self.explainer = ExplanationService()

    def ensure_candidate_embedding(self, candidate: Candidate) -> None:
        if candidate.embedding is not None:
            return
        text = candidate.parsed_profile.get("raw_text") if candidate.parsed_profile else None
        if not text:
            raise RuntimeError("Candidate has no parsed text to embed")
        embedding = embed_texts([summarize_text(text)])[0]
        candidate.embedding = embedding
        self.session.add(candidate)
        self.session.flush()

    def rerank(self, candidate: Candidate, jobs: list[Job]) -> list[float]:
        candidate_text = candidate.parsed_profile.get("raw_text", "") if candidate.parsed_profile else ""
        pairs = [(candidate_text, job.description or "") for job in jobs]
        results = self.reranker.score(pairs)
        return [result.score for result in results]

    def explain(self, candidate: Candidate, job: Job) -> str:
        resume_text = candidate.parsed_profile.get("raw_text", "") if candidate.parsed_profile else ""
        job_text = job.description or ""
        resume_spans = chunk_with_labels(resume_text, "R")
        job_spans = chunk_with_labels(job_text, "J")
        return self.explainer.explain(
            resume_spans=resume_spans,
            job_spans=job_spans,
            must_haves=job.must_have_skills or [],
            job_title=job.title,
        )

    def build_matches(self, candidate: Candidate, top_k: int = 5) -> Iterator[dict]:
        import json
        import logging

        logger = logging.getLogger(__name__)
        start_total = time.monotonic()

        try:
            print(f"DEBUG: Building matches for candidate {candidate.id}", flush=True)

            yield {"status": "Generating search queries...", "progress": 5}
            from services.shared.resume import generate_search_queries

            candidate_text = candidate.parsed_profile.get("raw_text", "")
            candidate_skills = candidate.skills or []
            queries = generate_search_queries(candidate_text, candidate_skills)

            if time.monotonic() - start_total > MATCH_TOTAL_BUDGET_S:
                yield {"status": "Timed out early (query gen).", "progress": 100, "data": []}
                return

            yield {"status": "Embedding search queries...", "progress": 10}
            query_embeddings = embed_texts(queries, task_type="RETRIEVAL_QUERY")

            yield {"status": "Retrieving jobs from vector database...", "progress": 15}
            all_retrieved: dict[int, tuple[Job, float]] = {}

            for i, embedding in enumerate(query_embeddings):
                vector = embedding
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()

                distance = func.cosine_distance(Job.embedding, cast(vector, Vector(_settings.embed_dim)))
                stmt = (
                    select(Job, (1 - distance).label("similarity"))
                    .where(Job.embedding.is_not(None))
                    .order_by(distance.asc())
                    .limit(25)
                )
                rows = self.session.execute(stmt).all()

                print(f"DEBUG: Query '{queries[i]}' retrieved {len(rows)} jobs", flush=True)

                for row in rows:
                    job, score = row[0], float(row[1])
                    if job.id not in all_retrieved or score > all_retrieved[job.id][1]:
                        all_retrieved[job.id] = (job, score)

                if time.monotonic() - start_total > MATCH_TOTAL_BUDGET_S:
                    break

            jobs = [item[0] for item in all_retrieved.values()]
            retrieval_scores = [item[1] for item in all_retrieved.values()]

            print(f"DEBUG: Total unique jobs after merge: {len(jobs)}", flush=True)

            if not jobs:
                yield {"status": "No jobs found.", "progress": 100, "data": []}
                return

            yield {"status": "Reranking candidates...", "progress": 25}
            rerank_scores = self.rerank(candidate, list(jobs))
            scored = list(zip(jobs, retrieval_scores, rerank_scores))

            candidates_to_check = scored[:MAX_JOBS_TO_CHECK]
            total_to_check = len(candidates_to_check)

            def process_match(job: Job, retrieval_score: float, rerank_score: float):
                try:
                    explanation_json = self.explain(candidate, job)
                    return job, retrieval_score, rerank_score, explanation_json
                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {e}", exc_info=True)
                    return job, retrieval_score, rerank_score, None

            results: list[MatchResult] = []

            yield {"status": f"Analyzing matches (0/{total_to_check})...", "progress": 30}

            start_explain = time.monotonic()
            checked = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=EXPLAIN_WORKERS) as executor:
                it = iter(candidates_to_check)
                pending: set[concurrent.futures.Future] = set()

                def submit_next():
                    nonlocal pending
                    try:
                        job, ret, rer = next(it)
                    except StopIteration:
                        return False
                    pending.add(executor.submit(process_match, job, ret, rer))
                    return True

                for _ in range(EXPLAIN_WORKERS):
                    if not submit_next():
                        break

                while pending:
                    elapsed_total = time.monotonic() - start_total
                    elapsed_explain = time.monotonic() - start_explain
                    if elapsed_total > MATCH_TOTAL_BUDGET_S or elapsed_explain > EXPLAIN_STAGE_TIMEOUT_S:
                        break

                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    if not done:
                        continue

                    for future in done:
                        checked += 1
                        current_progress = 30 + int((checked / max(total_to_check, 1)) * 65)
                        yield {
                            "status": f"Analyzing matches ({checked}/{total_to_check})...",
                            "progress": min(current_progress, 95),
                        }

                        try:
                            job, retrieval_score, rerank_score, explanation_json = future.result()
                            if not explanation_json:
                                continue

                            try:
                                explanation_data = json.loads(explanation_json)
                                summary = explanation_data.get("summary", "No summary available.")
                                llm_confidence = explanation_data.get("confidence")
                                reason_codes = explanation_data.get("reasons", [])
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse explanation JSON", exc_info=True)
                                summary = explanation_json
                                llm_confidence = None
                                reason_codes = []

                            if llm_confidence is None or llm_confidence < MIN_LLM_CONFIDENCE:
                                continue

                            confidence = llm_confidence

                            match = (
                                self.session.query(Match)
                                .filter(Match.candidate_id == candidate.id, Match.job_id == job.id)
                                .one_or_none()
                            )
                            if match is None:
                                match = Match(candidate_id=candidate.id, job_id=job.id)

                            match.retrieval_score = retrieval_score
                            match.rerank_score = rerank_score
                            match.confidence = confidence
                            match.explanation = summary
                            match.reason_codes = reason_codes
                            self.session.add(match)

                            results.append(
                                MatchResult(
                                    job=job,
                                    retrieval_score=retrieval_score,
                                    rerank_score=rerank_score,
                                    confidence=confidence,
                                    explanation=summary,
                                    reason_codes=reason_codes,
                                )
                            )

                            # Early exit once we have enough good matches
                            if len(results) >= top_k:
                                pending.clear()
                                break
                        except Exception as e:
                            logger.error(f"Exception in worker thread: {e}", exc_info=True)

                    # Keep the worker pool full
                    while len(pending) < EXPLAIN_WORKERS:
                        if not submit_next():
                            break

                # Best effort cancel (only cancels futures not started)
                for f in pending:
                    f.cancel()

            results.sort(key=lambda x: x.confidence or 0, reverse=True)
            results = results[:top_k]

            yield {"status": "Complete", "progress": 100, "data": [r.model_dump() for r in results]}
        except Exception as e:
            logger.error(f"Error building matches: {e}", exc_info=True)
            raise


def match_candidate(session: Session, candidate_id: uuid.UUID, limit: int = 5) -> Iterator[dict]:
    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        raise ValueError("Candidate not found")
    service = MatchService(session)
    return service.build_matches(candidate, top_k=limit)
