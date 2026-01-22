"""Candidate/job matching pipeline."""
from __future__ import annotations

import os
import time
import uuid
from typing import Iterator

import concurrent.futures
import json

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


# Pool sizes
RETRIEVAL_K_PER_QUERY = _env_int("RETRIEVAL_K_PER_QUERY", 40)     # how many per query
RERANK_POOL_SIZE = _env_int("RERANK_POOL_SIZE", 120)              # rerank this many
LLM_SCORE_POOL_SIZE = _env_int("LLM_SCORE_POOL_SIZE", 40)         # LLM-score this many

# Concurrency and time
LLM_WORKERS = _env_int("LLM_WORKERS", 4)
SCORE_STAGE_TIMEOUT_S = _env_float("SCORE_STAGE_TIMEOUT_S", 90.0)
TOTAL_BUDGET_S = _env_float("TOTAL_BUDGET_S", 240.0)

# Weighting
W_RETRIEVAL = _env_float("W_RETRIEVAL", 0.65)
W_RERANK = _env_float("W_RERANK", 0.35)
W_LLM = _env_float("W_LLM", 0.45)            # how much LLM influences final rank
MIN_LLM_CONFIDENCE = _env_float("MIN_LLM_CONFIDENCE", 0.40)  # softer than 0.7


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

    def _resume_spans(self, candidate: Candidate) -> str:
        resume_text = candidate.parsed_profile.get("raw_text", "") if candidate.parsed_profile else ""
        return chunk_with_labels(resume_text, "R")

    def _job_spans(self, job: Job) -> str:
        job_text = job.description or ""
        return chunk_with_labels(job_text, "J")

    def build_matches(self, candidate: Candidate, top_k: int = 5) -> Iterator[dict]:
        import logging

        logger = logging.getLogger(__name__)
        t0 = time.monotonic()

        try:
            self.ensure_candidate_embedding(candidate)

            yield {"status": "Generating search queries...", "progress": 5}
            from services.shared.resume import generate_search_queries

            candidate_text = candidate.parsed_profile.get("raw_text", "")
            candidate_skills = candidate.skills or []
            queries = generate_search_queries(candidate_text, candidate_skills)

            if time.monotonic() - t0 > TOTAL_BUDGET_S:
                yield {"status": "Timed out early.", "progress": 100, "data": []}
                return

            yield {"status": "Embedding search queries...", "progress": 10}
            query_embeddings = embed_texts(queries, task_type="RETRIEVAL_QUERY")

            yield {"status": "Retrieving jobs from vector database...", "progress": 15}
            # job_id -> (job, best_similarity)
            best_by_job: dict[int, tuple[Job, float]] = {}

            for i, embedding in enumerate(query_embeddings):
                vector = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                distance = func.cosine_distance(Job.embedding, cast(vector, Vector(_settings.embed_dim)))
                stmt = (
                    select(Job, (1 - distance).label("similarity"))
                    .where(Job.embedding.is_not(None))
                    .order_by(distance.asc())
                    .limit(RETRIEVAL_K_PER_QUERY)
                )
                rows = self.session.execute(stmt).all()

                for job, sim in rows:
                    sim_f = float(sim)
                    if job.id not in best_by_job or sim_f > best_by_job[job.id][1]:
                        best_by_job[job.id] = (job, sim_f)

                if time.monotonic() - t0 > TOTAL_BUDGET_S:
                    break

            if not best_by_job:
                yield {"status": "No jobs found.", "progress": 100, "data": []}
                return

            # IMPORTANT: sort by similarity DESC so we actually evaluate the best ones
            retrieved_sorted = sorted(best_by_job.values(), key=lambda x: x[1], reverse=True)

            # Rerank a large pool
            pool = retrieved_sorted[:RERANK_POOL_SIZE]
            pool_jobs = [j for j, _ in pool]
            pool_retrieval = [s for _, s in pool]

            yield {"status": "Reranking candidates...", "progress": 25}
            pool_rerank = self.rerank(candidate, pool_jobs)

            combined = []
            for job, ret, rer in zip(pool_jobs, pool_retrieval, pool_rerank):
                combo = (W_RETRIEVAL * ret) + (W_RERANK * rer)
                combined.append((job, ret, rer, combo))

            # Sort by combined score DESC
            combined.sort(key=lambda x: x[3], reverse=True)

            # LLM score only the best subset
            to_score = combined[:LLM_SCORE_POOL_SIZE]
            total_to_score = len(to_score)

            resume_spans = self._resume_spans(candidate)

            def score_one(job: Job, ret: float, rer: float, combo: float):
                try:
                    score_json = self.explainer.score(
                        resume_spans=resume_spans,
                        job_spans=self._job_spans(job),
                        must_haves=job.must_have_skills or [],
                        job_title=job.title,
                    )
                    data = json.loads(score_json)
                    llm_conf = data.get("confidence")
                    reasons = data.get("reasons", [])
                    return job, ret, rer, combo, llm_conf, reasons
                except Exception as e:
                    logger.error(f"LLM score failed for job {job.id}: {e}", exc_info=True)
                    return job, ret, rer, combo, None, []

            yield {"status": f"Scoring top roles (0/{total_to_score})...", "progress": 35}

            scored_rows = []
            start_score = time.monotonic()
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
                futures = [ex.submit(score_one, j, r, rr, c) for (j, r, rr, c) in to_score]
                for f in concurrent.futures.as_completed(futures):
                    completed += 1
                    yield {
                        "status": f"Scoring top roles ({completed}/{total_to_score})...",
                        "progress": min(35 + int((completed / max(total_to_score, 1)) * 35), 70),
                    }
                    if time.monotonic() - start_score > SCORE_STAGE_TIMEOUT_S:
                        break
                    if time.monotonic() - t0 > TOTAL_BUDGET_S:
                        break
                    scored_rows.append(f.result())

            # Combine: use LLM as ranking signal (not a strict gate)
            ranked = []
            for job, ret, rer, combo, llm_conf, reasons in scored_rows:
                llm_val = float(llm_conf) if isinstance(llm_conf, (int, float)) else 0.0
                if llm_val < MIN_LLM_CONFIDENCE:
                    # keep it, but down-weight hard instead of dropping
                    llm_val *= 0.5

                final_score = (1 - W_LLM) * combo + (W_LLM) * llm_val
                ranked.append((job, ret, rer, llm_val, final_score, reasons))

            ranked.sort(key=lambda x: x[4], reverse=True)

            # Now generate full explanations only for top_k
            results: list[MatchResult] = []

            yield {"status": "Generating final explanations...", "progress": 80}

            for idx, (job, ret, rer, llm_val, final_score, score_reasons) in enumerate(ranked[:top_k], start=1):
                if time.monotonic() - t0 > TOTAL_BUDGET_S:
                    break

                explanation_json = self.explainer.explain(
                    resume_spans=resume_spans,
                    job_spans=self._job_spans(job),
                    must_haves=job.must_have_skills or [],
                    job_title=job.title,
                )
                try:
                    exp = json.loads(explanation_json)
                    summary = exp.get("summary", "No summary available.")
                    exp_conf = exp.get("confidence", llm_val)
                    reason_codes = exp.get("reasons", score_reasons)
                except Exception:
                    summary = "No summary available."
                    exp_conf = llm_val
                    reason_codes = score_reasons

                match = (
                    self.session.query(Match)
                    .filter(Match.candidate_id == candidate.id, Match.job_id == job.id)
                    .one_or_none()
                )
                if match is None:
                    match = Match(candidate_id=candidate.id, job_id=job.id)

                match.retrieval_score = float(ret)
                match.rerank_score = float(rer)
                match.confidence = float(exp_conf) if exp_conf is not None else float(final_score)
                match.explanation = summary
                match.reason_codes = reason_codes
                self.session.add(match)

                results.append(
                    MatchResult(
                        job=job,
                        retrieval_score=float(ret),
                        rerank_score=float(rer),
                        confidence=float(exp_conf) if exp_conf is not None else float(final_score),
                        explanation=summary,
                        reason_codes=reason_codes,
                    )
                )

                yield {"status": f"Finalizing ({idx}/{min(top_k, len(ranked))})...", "progress": 80 + int((idx / max(top_k, 1)) * 19)}

            results.sort(key=lambda x: x.confidence or 0, reverse=True)
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
