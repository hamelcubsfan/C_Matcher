import sys
import logging
from sqlalchemy import select
from services.shared.db import get_session_factory
from services.shared.models import Job, Candidate
from services.shared.embeddings import embed_texts
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def debug_matches():
    session = get_session_factory()()
    try:
        # 1. Fetch the candidate (assuming only one due to ephemeral policy)
        candidate = session.query(Candidate).first()
        if not candidate:
            logger.error("No candidate found in DB.")
            return

        logger.info(f"Candidate: {candidate.full_name}")
        logger.info(f"Candidate Skills: {candidate.skills[:5]}...")

        # 2. Search for jobs by title since IDs seem wrong
        logger.info("Searching for jobs with 'Recruit' or 'Sourc' in title...")
        jobs = session.query(Job).filter(
            (Job.title.ilike("%Recruit%")) | (Job.title.ilike("%Sourc%"))
        ).all()
        
        if not jobs:
            logger.error("No recruiting jobs found in DB.")
            return

        # 3. Calculate similarity
        logger.info(f"Found {len(jobs)} recruiting jobs.")
        for job in jobs:
            sim = cosine_similarity(candidate.embedding, job.embedding)
            logger.info(f"Job {job.greenhouse_job_id}: {job.title}")
            logger.info(f"  Similarity: {sim:.4f}")
            logger.info(f"  Intern Filter Check: 'intern' in title? {'intern' in job.title.lower()}")

    finally:
        session.close()

if __name__ == "__main__":
    debug_matches()
