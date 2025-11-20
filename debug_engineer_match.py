import sys
import os
from sqlalchemy import select, func, cast
from pgvector.sqlalchemy import Vector
from services.shared.db import get_session_factory
from services.shared.models import Candidate, Job
from services.shared.config import get_settings

# Ensure we can import services
sys.path.append(os.getcwd())

def debug_matches():
    settings = get_settings()
    with get_session_factory()() as session:
        # Get the most recent candidate (Sumedh)
        candidate = session.query(Candidate).order_by(Candidate.id.desc()).first()
        if not candidate:
            print("No candidate found.")
            return

        print(f"Candidate: {candidate.full_name}")
        print("-" * 20)
        
        # Reconstruct the semantic summary to see what was embedded
        # Note: We can't get the exact string used for embedding from the DB unless we stored it,
        # but we can recreate the logic from main.py to see what it *would* be.
        # Or better, let's just look at the raw text and skills to see if they are parsed correctly.
        print(f"Parsed Skills: {candidate.skills}")
        
        # Check retrieval for specific keywords
        vector = candidate.embedding
        if vector is None:
            print("Candidate has no embedding.")
            return
            
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        # Search for "Perception" or "Robotics" jobs
        print("\nSearching for 'Perception' or 'Robotics' jobs...")
        stmt = (
            select(Job, (1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).label("similarity"))
            .where(Job.title.ilike("%Perception%") | Job.title.ilike("%Robotics%") | Job.title.ilike("%Computer Vision%"))
            .order_by((1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).desc())
            .limit(10)
        )
        
        rows = session.execute(stmt).all()
        for row in rows:
            job = row[0]
            score = row[1]
            print(f"[{score:.4f}] {job.title} (ID: {job.id})")

        # Also check the top matches generally to see what IS matching
        print("\nTop 10 General Matches from DB (with LLM Confidence):")
        stmt = (
            select(Job, (1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).label("similarity"))
            .order_by((1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).desc())
            .limit(10)
        )
        rows = session.execute(stmt).all()
        
        from services.match.service import MatchService
        match_service = MatchService(session)
        import json
        
        for row in rows:
            job = row[0]
            score = row[1]
            print(f"[{score:.4f}] {job.title} (ID: {job.id})")
            
            # Run explanation to get confidence
            try:
                explanation_json = match_service.explain(candidate, job)
                data = json.loads(explanation_json)
                confidence = data.get("confidence")
                print(f"    -> LLM Confidence: {confidence}")
                print(f"    -> Reason: {data.get('summary')[:100]}...")
            except Exception as e:
                print(f"    -> LLM Failed: {e}")

if __name__ == "__main__":
    debug_matches()
