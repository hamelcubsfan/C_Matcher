"""Candidate/job matching pipeline."""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Iterator, Optional, Tuple, List

from sqlalchemy import cast, func, select, text
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


# Retrieval sizes
VECTOR_K_PER_QUERY = _env_int("VECTOR_K_PER_QUERY", 60)
VECTOR_QUERY_LIMIT = _env_int("VECTOR_QUERY_LIMIT", 6)  # how many query vectors to run
LEXICAL_LIMIT = _env_int("LEXICAL_LIMIT", 250)

# Pool sizes
RERANK_POOL_SIZE = _env_int("RERANK_POOL_SIZE", 160)
LLM_SCORE_POOL_SIZE = _env_int("LLM_SCORE_POOL_SIZE", 50)

# Concurrency and budgets
LLM_WORKERS = _env_int("LLM_WORKERS", 4)
SCORE_STAGE_TIMEOUT_S = _env_float("SCORE_STAGE_TIMEOUT_S", 90.0)
TOTAL_BUDGET_S = _env_float("TOTAL_BUDGET_S", 240.0)

# Weighting
W_VECTOR = _env_float("W_VECTOR", 0.60)
W_RERANK = _env_float("W_RERANK", 0.25)
W_LEXICAL = _env_float("W_LEXICAL", 0.15)
W_LLM = _env_float("W_LLM", 0.45)

# Confidence handling
MIN_LLM_CONFIDENCE = _env_float("MIN_LLM_CONFIDENCE", 0.35)  # not a hard filter
LOW_CONF_PENALTY_MULT = _env_float("LOW_CONF_PENALTY_MULT", 0.55)


def _now() -> float:
    return time.monotonic()


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _build_sensor_adjacent_queries(resume_text: str, skills: List[str]) -> List[str]:
    """
    Heuristic recall: if the resume looks like CV/perception/edge ML,
    add a couple adjacent queries that surface sensor validation, calibration, fusion roles.
    """
    blob = (resume_text or "").lower()
    skills_blob = " ".join([s.lower() for s in (skills or [])])

    cv_signals = [
        "computer vision",
        "perception",
        "3d",
        "multi-view",
        "multiview",
        "tracking",
        "kalman",
        "tensorrt",
        "jetson",
        "tflite",
        "onnx",
        "cuda",
        "clip",
        "vlm",
        "simclr",
        "byol",
    ]
    is_cv = any(sig in blob or sig in skills_blob for sig in cv_signals)

    if not is_cv:
        return []

    return [
        "Senior Machine Learning Engineer sensor validation calibration alignment sensor health fault detection",
        "Machine learning sensor fusion lidar radar camera calibration validation perception health monitoring",
    ]


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

    def _resume_spans(self, candidate: Candidate) -> str:
        resume_text = candidate.parsed_profile.get("raw_text", "") if candidate.parsed_profile else ""
        return chunk_with_labels(resume_text, "R")

    def _job_spans(self, job: Job) -> str:
        job_text = job.description or ""
        return chunk_with_labels(job_text, "J")

    def rerank(self, candidate: Candidate, jobs: list[Job]) -> list[float]:
        candidate_text = candidate.parsed_profile.get("raw_text", "") if candidate.parsed_profile else ""
        pairs = [(candidate_text, job.description or "") for job in jobs]
        results = self.reranker.score(pairs)
        return [result.score for result in results]

    def _vector_retrieve(self, vectors: list, k: int) -> dict[int, Tuple[Job, float]]:
        """
        Returns best similarity per job id using cosine_distance.
        """
        best_by_job: dict[int, Tuple[Job, float]] = {}

        for vec in vectors:
            vector = vec.tolist() if hasattr(vec, "tolist") else vec
            distance = func.cosine_distance(Job.embedding, cast(vector, Vector(_settings.embed_dim)))
            stmt = (
                select(Job, (1 - distance).label("similarity"))
                .where(Job.embedding.is_not(None))
                .order_by(distance.asc())
                .limit(k)
            )
            rows = self.session.execute(stmt).all()
            for job, sim in rows:
                sim_f = float(sim)
                if job.id not in best_by_job or sim_f > best_by_job[job.id][1]:
                    best_by_job[job.id] = (job, sim_f)

        return best_by_job

    def _lexical_recall(self, resume_text: str, skills: list[str]) -> dict[int, float]:
        """
        Full-text recall and targeted keyword recall.
        Produces a lexical score mapped to 0..1 for blending.
        """
        # Build a compact query string from skills + a few resume tokens
        top_skills = [s.strip() for s in (skills or []) if s and len(s) <= 40][:12]

        # Always include a sensor-adjacent clause for CV candidates
        extra_terms = _build_sensor_adjacent_queries(resume_text, top_skills)
        extra_tokens = []
        for q in extra_terms:
            extra_tokens.extend(q.split())

        tokens = top_skills + extra_tokens
        tokens = [t for t in tokens if t]
        # De-dup, keep order
        seen = set()
        tokens_unique = []
        for t in tokens:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                tokens_unique.append(t)

        # Keep query short so it stays stable
        query_text = " ".join(tokens_unique[:25]).strip()
        if not query_text:
            query_text = "machine learning computer vision"

        # FTS uses plainto_tsquery which is forgiving
        # We rank using ts_rank_cd and normalize
        fts_sql = text(
            """
            select
              id,
              ts_rank_cd(
                setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
                setweight(to_tsvector('english', coalesce(description,'')), 'B'),
                plainto_tsquery('english', :q)
              ) as rank
            from jobs
            where
              (
                setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
                setweight(to_tsvector('english', coalesce(description,'')), 'B')
              ) @@ plainto_tsquery('english', :q)
            order by rank desc
            limit :lim
            """
        )

        rows = self.session.execute(fts_sql, {"q": query_text, "lim": LEXICAL_LIMIT}).all()

        # Normalize ranks into 0..1 by dividing by max rank in this batch
        lexical: dict[int, float] = {}
        if not rows:
            return lexical

        max_rank = max(float(r[1]) for r in rows) or 1.0
        for job_id, rank in rows:
            lexical[int(job_id)] = _clamp01(float(rank) / max_rank)

        # Targeted keyword recall as a safety net (very useful for sensor roles)
        # Adds jobs even if FTS missed them due to tokenization.
        keyword_re = (
            "sensor|validation|calibration|alignment|fusion|radar|lidar|fault|health|positioning"
        )
        kw_sql = text(
            """
            select id
            from jobs
            where
              lower(coalesce(title,'')) ~ :re
              or lower(coalesce(description,'')) ~ :re
            limit 400
            """
        )
        kw_rows = self.session.execute(kw_sql, {"re": keyword_re}).all()
        for (job_id,) in kw_rows:
            jid = int(job_id)
            # Give keyword-only hits a small baseline
            if jid not in lexical:
                lexical[jid] = 0.30

        return lexical

    def _llm_score_job(
        self,
        resume_spans: str,
        job: Job,
    ) -> Tuple[Optional[float], list]:
        """
        Returns (llm_confidence, reasons_list)
        Uses explainer.score() if available, else falls back to explainer.explain().
        """
        must_haves = job.must_have_skills or []

        if hasattr(self.explainer, "score"):
            try:
                score_json = self.explainer.score(
                    resume_spans=resume_spans,
                    job_spans=self._job_spans(job),
                    must_haves=must_haves,
                    job_title=job.title,
                )
                data = _safe_json_loads(score_json)
                conf = data.get("confidence")
                reasons = data.get("reasons", [])
                if isinstance(conf, (int, float)):
                    return float(conf), reasons
                return None, reasons
            except Exception:
                return None, []

        # Fallback: use explain() and parse confidence
        try:
            exp_json = self.explainer.explain(
                resume_spans=resume_spans,
                job_spans=self._job_spans(job),
                must_haves=must_haves,
                job_title=job.title,
            )
            data = _safe_json_loads(exp_json)
            conf = data.get("confidence")
            reasons = data.get("reasons", [])
            if isinstance(conf, (int, float)):
                return float(conf), reasons
            return None, reasons
        except Exception:
            return None, []

    def build_matches(self, candidate: Candidate, top_k: int = 5) -> Iterator[dict]:
        import concurrent.futures
        import logging

        logger = logging.getLogger(__name__)
        start = _now()

        try:
            self.ensure_candidate_embedding(candidate)
            resume_text = candidate.parsed_profile.get("raw_text", "") if candidate.parsed_profile else ""
            skills = candidate.skills or []

            yield {"status": "Generating search queries...", "progress": 5}
            from services.shared.resume import generate_search_queries

            queries = generate_search_queries(resume_text, skills)

            # Add sensor-adjacent recall queries for CV candidates
            queries.extend(_build_sensor_adjacent_queries(resume_text, skills))

            # Always include the full resume summary as a retrieval query
            queries.append(summarize_text(resume_text))

            # Keep query list bounded
            queries = [q for q in queries if q and q.strip()]
            queries = queries[: max(VECTOR_QUERY_LIMIT, 1)]

            if _now() - start > TOTAL_BUDGET_S:
                yield {"status": "Timed out early.", "progress": 100, "data": []}
                return

            yield {"status": "Embedding search queries...", "progress": 10}
            query_embeddings = embed_texts(queries, task_type="RETRIEVAL_QUERY")

            # Always include candidate embedding as an additional vector
            query_embeddings.append(candidate.embedding)

            yield {"status": "Retrieving jobs (vector + lexical)...", "progress": 15}
            best_by_job = self._vector_retrieve(query_embeddings, VECTOR_K_PER_QUERY)

            lexical_scores = self._lexical_recall(resume_text, skills)

            # Merge lexical-only jobs into pool with a modest vector seed
            # This guarantees adjacent roles are evaluated downstream.
            for job_id, lex_score in lexical_scores.items():
                if job_id not in best_by_job:
                    job = self.session.get(Job, job_id)
                    if job is None:
                        continue
                    # Seed vector similarity low but non-zero so it gets into the pool
                    best_by_job[job_id] = (job, 0.45)

            if not best_by_job:
                yield {"status": "No jobs found.", "progress": 100, "data": []}
                return

            # Build pool with (job, vector_sim, lexical_sim)
            merged: list[tuple[Job, float, float]] = []
            for job_id, (job, vec_sim) in best_by_job.items():
                lex = float(lexical_scores.get(job_id, 0.0))
                merged.append((job, float(vec_sim), lex))

            # Sort by blended pre-score to choose a big rerank pool
            merged.sort(key=lambda x: (W_VECTOR * x[1]) + (W_LEXICAL * x[2]), reverse=True)

            # Rerank pool
            pool = merged[:RERANK_POOL_SIZE]
            pool_jobs = [j for (j, _, _) in pool]
            pool_vec = [v for (_, v, _) in pool]
            pool_lex = [l for (_, _, l) in pool]

            yield {"status": "Reranking (fast)...", "progress": 25}
            pool_rer = self.rerank(candidate, pool_jobs)

            combined_rows = []
            for job, vec_sim, lex_sim, rer in zip(pool_jobs, pool_vec, pool_lex, pool_rer):
                base = (W_VECTOR * vec_sim) + (W_LEXICAL * lex_sim) + (W_RERANK * float(rer))
                combined_rows.append((job, vec_sim, lex_sim, float(rer), base))

            combined_rows.sort(key=lambda x: x[4], reverse=True)

            # LLM score subset
            to_score = combined_rows[:LLM_SCORE_POOL_SIZE]
            total_to_score = len(to_score)

            resume_spans = self._resume_spans(candidate)

            yield {"status": f"Scoring top roles ({0}/{total_to_score})...", "progress": 35}

            scored: list[tuple[Job, float, float, float, float, float, list]] = []
            score_start = _now()
            completed = 0

            def score_one(row):
                job, vec_sim, lex_sim, rer, base = row
                llm_conf, reasons = self._llm_score_job(resume_spans, job)
                llm_val = float(llm_conf) if isinstance(llm_conf, (int, float)) else 0.0
                return job, vec_sim, lex_sim, rer, base, llm_val, reasons

            with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
                futures = [ex.submit(score_one, r) for r in to_score]
                for f in concurrent.futures.as_completed(futures):
                    completed += 1
                    if _now() - score_start > SCORE_STAGE_TIMEOUT_S:
                        break
                    if _now() - start > TOTAL_BUDGET_S:
                        break

                    job, vec_sim, lex_sim, rer, base, llm_val, reasons = f.result()

                    # Penalize very low LLM confidence, but do not drop
                    if llm_val < MIN_LLM_CONFIDENCE:
                        llm_val *= LOW_CONF_PENALTY_MULT

                    final_score = (1 - W_LLM) * base + (W_LLM) * llm_val
                    scored.append((job, vec_sim, lex_sim, rer, base, final_score, reasons))

                    yield {
                        "status": f"Scoring top roles ({completed}/{total_to_score})...",
                        "progress": min(35 + int((completed / max(total_to_score, 1)) * 35), 70),
                    }

            if not scored:
                # Fallback if scoring stage got cut short
                scored = [(job, vec, lex, rer, base, base, []) for (job, vec, lex, rer, base) in combined_rows[:top_k]]

            scored.sort(key=lambda x: x[5], reverse=True)

            # Final explanations only for top_k
            yield {"status": "Generating final explanations...", "progress": 80}

            results: list[MatchResult] = []

            for idx, (job, vec_sim, lex_sim, rer, base, final_score, score_reasons) in enumerate(
                scored[:top_k], start=1
            ):
                if _now() - start > TOTAL_BUDGET_S:
                    break

                exp_json = self.explainer.explain(
                    resume_spans=resume_spans,
                    job_spans=self._job_spans(job),
                    must_haves=job.must_have_skills or [],
                    job_title=job.title,
                )

                exp = _safe_json_loads(exp_json)
                summary = exp.get("summary", "No summary available.")
                exp_conf = exp.get("confidence")
                reasons = exp.get("reasons", score_reasons)

                confidence = None
                if isinstance(exp_conf, (int, float)):
                    confidence = float(exp_conf)
                else:
                    confidence = float(final_score)

                match = (
                    self.session.query(Match)
                    .filter(Match.candidate_id == candidate.id, Match.job_id == job.id)
                    .one_or_none()
                )
                if match is None:
                    match = Match(candidate_id=candidate.id, job_id=job.id)

                match.retrieval_score = float(vec_sim)
                match.rerank_score = float(rer)
                match.confidence = confidence
                match.explanation = summary
                match.reason_codes = reasons
                self.session.add(match)

                results.append(
                    MatchResult(
                        job=job,
                        retrieval_score=float(vec_sim),
                        rerank_score=float(rer),
                        confidence=confidence,
                        explanation=summary,
                        reason_codes=reasons,
                    )
                )

                yield {
                    "status": f"Finalizing ({idx}/{min(top_k, len(scored))})...",
                    "progress": 80 + int((idx / max(top_k, 1)) * 19),
                }

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
