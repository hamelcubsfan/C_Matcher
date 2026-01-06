"""Candidate/job matching pipeline."""
from __future__ import annotations

import uuid

from sqlalchemy import func, select, cast
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

try:  # pragma: no cover - optional dependency surface
    from pgvector.sqlalchemy import CosineDistance  # type: ignore
except ImportError:  # Render's pgvector build omits this helper
    CosineDistance = None  # type: ignore

from services.explain.service import ExplanationService
from services.match.rerank import get_reranker
from services.shared.config import get_settings
from services.shared.embeddings import embed_texts
from services.shared.models import Candidate, Job, Match
from services.shared.resume import chunk_with_labels, summarize_text
from services.shared.schemas import MatchResult

_settings = get_settings()

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

    def retrieve_jobs(self, candidate: Candidate, limit: int = 10) -> list[tuple[Job, float]]:
        vector = candidate.embedding
        if vector is None:
            raise RuntimeError("Candidate embedding missing")
        
        # Ensure vector is a list, not a numpy array, for psycopg adaptation
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
            
        print(f"DEBUG: Vector start: {vector[:5]}", flush=True)
            
        # Force explicit cast and function call to match debug script
        # if CosineDistance is not None:
        #     distance = CosineDistance(Job.embedding, vector)
        # else:
        # Explicitly cast the list to a Vector type so Postgres knows which function to use
        distance = func.cosine_distance(Job.embedding, cast(vector, Vector(_settings.embed_dim)))
        stmt = (
            select(Job, (1 - distance).label("similarity"))
            # .where(Job.posting_status == "open")
            .where(Job.embedding.is_not(None))
            .order_by(distance.asc())
            .limit(50)
        )
        rows = self.session.execute(stmt).all()
        
        # DEBUG: Print raw retrieval results
        print(f"DEBUG: Raw retrieval count: {len(rows)}", flush=True)
        for i, row in enumerate(rows[:5]):
            print(f"DEBUG: Raw Rank {i+1}: Job {row[0].id} ({row[0].title}) - Sim: {row[1]:.4f}", flush=True)
            
        return [(row[0], float(row[1])) for row in rows]

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

    def build_matches(self, candidate: Candidate, top_k: int = 5) -> list[MatchResult]:
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            print(f"DEBUG: Building matches for candidate {candidate.id}", flush=True)
            
            # 1. Query Expansion: Get diverse search angles
            yield {"status": "Generating search queries...", "progress": 5}
            from services.shared.resume import generate_search_queries
            candidate_text = candidate.parsed_profile.get("raw_text", "")
            candidate_skills = candidate.skills or []
            queries = generate_search_queries(candidate_text, candidate_skills)
            
            # 2. Embed all queries
            yield {"status": "Embedding search queries...", "progress": 10}
            query_embeddings = embed_texts(queries)
            
            # 3. Multi-Vector Retrieval
            yield {"status": "Retrieving jobs from vector database...", "progress": 15}
            all_retrieved: dict[int, tuple[Job, float]] = {}
            
            for i, embedding in enumerate(query_embeddings):
                # Create a temporary candidate object for retrieval (hacky but works with existing method)
                # Or better, refactor retrieve_jobs to take a vector. 
                # For now, let's just call the lower-level logic directly or update retrieve_jobs.
                # Let's update retrieve_jobs to take an optional vector override.
                
                # Actually, let's just do the search here to avoid breaking the interface yet
                vector = embedding
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                
                distance = func.cosine_distance(Job.embedding, cast(vector, Vector(_settings.embed_dim)))
                stmt = (
                    select(Job, (1 - distance).label("similarity"))
                    .where(Job.embedding.is_not(None))
                    .order_by(distance.asc())
                    .limit(25) # Fetch top 25 for EACH query
                )
                rows = self.session.execute(stmt).all()
                
                print(f"DEBUG: Query '{queries[i]}' retrieved {len(rows)} jobs", flush=True)
                
                for row in rows:
                    job, score = row[0], float(row[1])
                    # Keep the HIGHEST score if found multiple times
                    if job.id not in all_retrieved or score > all_retrieved[job.id][1]:
                        all_retrieved[job.id] = (job, score)

            jobs = [item[0] for item in all_retrieved.values()]
            retrieval_scores = [item[1] for item in all_retrieved.values()]
            
            print(f"DEBUG: Total unique jobs after merge: {len(jobs)}", flush=True)
            
            if not jobs:
                print("DEBUG: No jobs retrieved from DB", flush=True)
                yield {"status": "No jobs found.", "progress": 100, "data": []}
                return

            yield {"status": "Reranking candidates...", "progress": 25}
            rerank_scores = self.rerank(candidate, list(jobs))
            
            scored = list(zip(jobs, retrieval_scores, rerank_scores))
            
            # DEBUG: Print rerank scores for top 5 vector matches
            for i in range(min(5, len(scored))):
                job, ret, rer = scored[i]
                print(f"DEBUG: Job {job.id} ({job.title}) - Retrieval: {ret:.4f}, Rerank: {rer:.4f}", flush=True)

            # CRITICAL FIX: The reranker is burying the Recruiter jobs. 
            # We trust the vector search (which put them at #1, #2, #3).
            # We will NOT sort by rerank score for the cutoff. We will keep retrieval order.
            # scored.sort(key=lambda x: x[2], reverse=True)
            
            # Don't truncate to top_k yet! We need to find top_k *valid* matches.
            # FOMO FIX: We will check MORE candidates (50) and NOT stop early.
            # We want to find the absolute best matches in the pool, not just the first 5 good ones.
            candidates_to_check = scored[:50]
            total_to_check = len(candidates_to_check)
            
            import json
            
            # Parallelize explanation generation
            import concurrent.futures
            
            # Helper function for parallel execution
            def process_match(job, retrieval_score, rerank_score):
                try:
                    explanation_json = self.explain(candidate, job)
                    return job, retrieval_score, rerank_score, explanation_json
                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {e}", exc_info=True)
                    return job, retrieval_score, rerank_score, None

            results: list[MatchResult] = []
            
            # Use ThreadPoolExecutor with max_workers=8 to respect rate limits
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all tasks
                future_to_job = {
                    executor.submit(process_match, job, ret, rer): job 
                    for job, ret, rer in candidates_to_check
                }
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_job):
                    completed_count += 1
                    # Calculate progress from 30% to 95%
                    current_progress = 30 + int((completed_count / total_to_check) * 65)
                    yield {"status": f"Analyzing matches ({completed_count}/{total_to_check})...", "progress": current_progress}
                    
                    try:
                        job, retrieval_score, rerank_score, explanation_json = future.result()
                        
                        if not explanation_json:
                            continue

                        # Parse the JSON explanation to get structured data
                        try:
                            explanation_data = json.loads(explanation_json)
                            summary = explanation_data.get("summary", "No summary available.")
                            llm_confidence = explanation_data.get("confidence")
                            reason_codes = explanation_data.get("reasons", [])
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse explanation JSON", exc_info=True)
                            print(f"DEBUG: Failed to parse JSON for Job {job.id}", flush=True)
                            summary = explanation_json
                            llm_confidence = None
                            reason_codes = []

                        # Filter out low confidence matches based on LLM assessment
                        if llm_confidence is not None:
                            print(f"DEBUG: Job {job.id} ({job.title}): LLM confidence {llm_confidence}", flush=True)
                        else:
                            print(f"DEBUG: Job {job.id} ({job.title}): No LLM confidence returned", flush=True)

                        # Filter by LLM confidence
                        # User requested strict filtering: only return GOOD matches (> 0.7)
                        if llm_confidence is None or llm_confidence < 0.7:
                            print(f"DEBUG: Skipping match for job {job.id} due to low/missing LLM confidence: {llm_confidence}", flush=True)
                            continue

                        # Use LLM confidence if available, otherwise fall back to retrieval/rerank average
                        if llm_confidence is not None:
                            confidence = llm_confidence
                        else:
                            confidence = (retrieval_score + rerank_score) / 2 if rerank_score is not None else retrieval_score
                        
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
                        match.explanation = summary  # Store the clean text summary
                        match.reason_codes = reason_codes
                        self.session.add(match)
                        results.append(
                            MatchResult(
                                job=job,
                                retrieval_score=retrieval_score,
                                rerank_score=rerank_score,
                                confidence=confidence,
                                explanation=summary, # Return the clean text summary
                                reason_codes=reason_codes,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Exception in worker thread: {e}", exc_info=True)
            
            # Sort by confidence descending so the best matches are first
            results.sort(key=lambda x: x.confidence or 0, reverse=True)
            
            # NOW we truncate to top_k, ensuring we kept the BEST ones
            results = results[:top_k]
            
            logger.info(f"Returning {len(results)} matches")
            
            # Final yield with data
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
