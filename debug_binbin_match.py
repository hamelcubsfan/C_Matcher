import sys
import os
from sqlalchemy import select, func, cast
from pgvector.sqlalchemy import Vector

# Add current directory to path so we can import services
sys.path.append(os.getcwd())

from services.shared.db import get_session_factory
from services.shared.models import Candidate, Job
from services.shared.config import get_settings

def debug_binbin():
    settings = get_settings()
    with get_session_factory()() as session:
        # Find Binbin Li
        candidate = session.query(Candidate).filter(Candidate.full_name.ilike("%Binbin%")).first()
        if not candidate:
            print("Candidate Binbin Li not found!")
            return

        print(f"Candidate: {candidate.full_name} (ID: {candidate.id})")
        
        # Reconstruct semantic summary
        if candidate.parsed_profile:
            profile = candidate.parsed_profile
            experience = profile.get("experience", [])
            recent_role = experience[0] if experience else None
            recent_role_text = f"{recent_role.get('title')} at {recent_role.get('company')}" if recent_role else ""
            skills = profile.get("skills", [])
            
            semantic_summary = f"""
            Role: {recent_role_text}
            Summary: {profile.get('summary')}
            Top Skills: {', '.join(skills[:10])}
            """
            print("-" * 20)
            print(f"Reconstructed Semantic Summary:\n{semantic_summary}")
            print("-" * 20)
        
        vector = candidate.embedding
        if vector is None:
            print("No embedding found for candidate!")
            return

        # Target Jobs we expect to see
        target_ids = [168, 169, 129, 163, 60, 126]
        
        print(f"\nChecking specific target jobs:")
        for job_id in target_ids:
            job = session.get(Job, job_id)
            if not job:
                print(f"Job {job_id} not found in DB")
                continue
                
            # Calculate similarity manually
            # Note: pgvector cosine_distance returns 1 - cosine_similarity
            # We want similarity, so 1 - distance
            distance = session.scalar(
                select(func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim))))
                .where(Job.id == job_id)
            )
            similarity = 1 - distance if distance is not None else 0
            print(f"Job {job_id}: {job.title} - Similarity: {similarity:.4f}")

        # Get top 10 actual matches by vector similarity
        print(f"\nTop 10 Actual Vector Matches:")
        distance = func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))
        stmt = (
            select(Job, (1 - distance).label("similarity"))
            .where(Job.embedding.is_not(None))
            .order_by(distance.asc())
            .limit(10)
        )
        rows = session.execute(stmt).all()
        for i, row in enumerate(rows):
            print(f"Rank {i+1}: Job {row[0].id} ({row[0].title}) - Sim: {row[1]:.4f}")

if __name__ == "__main__":
    debug_binbin()
