import sys
import os
from sqlalchemy import select, func, cast
from pgvector.sqlalchemy import Vector
from services.shared.db import get_session_factory
from services.shared.models import Candidate, Job
from services.shared.config import get_settings

# Ensure we can import services
sys.path.append(os.getcwd())

def debug_recruiter_matches():
    settings = get_settings()
    with get_session_factory()() as session:
        # Get the specific candidate from the logs
        target_id = "4b9a7e00-5baa-4fb8-b5b8-4ad080035312"
        candidate = session.query(Candidate).filter(Candidate.id == target_id).first()
        if not candidate:
            print(f"Candidate {target_id} not found. Falling back to random.")
            candidate = session.query(Candidate).first()
        if not candidate:
            print("No candidate found.")
            return

        print(f"Candidate: {candidate.full_name} (ID: {candidate.id})")
        print("-" * 20)
        
        # Reconstruct semantic summary from parsed_profile
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
            print(f"Reconstructed Semantic Summary:\n{semantic_summary}")
        else:
            print("No parsed_profile found!")
            
        print("-" * 20)
        print(f"Parsed Skills: {candidate.skills}")
        
        vector = candidate.embedding
        if vector is None:
            print("Candidate has no embedding.")
            return
            
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
            
        print(f"DEBUG: Vector start: {vector[:5]}")

        # 1. Search for ACTUAL Recruiter jobs to see where they rank
        print("\nSearching for 'Recruiter', 'Talent', 'Sourcing' jobs...")
        stmt = (
            select(Job, (1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).label("similarity"))
            .where(Job.title.ilike("%Recruit%") | Job.title.ilike("%Talent%") | Job.title.ilike("%Sourc%"))
            .order_by((1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).desc())
            .limit(10)
        )
        
        rows = session.execute(stmt).all()
        if not rows:
            print("NO RECRUITER JOBS FOUND IN DB!")
        for row in rows:
            job = row[0]
            score = row[1]
            print(f"[{score:.4f}] {job.title} (ID: {job.id})")

        # 2. Check the Top 20 General Matches to see what is crowding them out
        print("\nTop 20 General Matches from DB (with posting_status='open'):")
        stmt = (
            select(Job, (1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).label("similarity"))
            .where(Job.posting_status == "open")
            .order_by((1 - func.cosine_distance(Job.embedding, cast(vector, Vector(settings.embed_dim)))).desc())
            .limit(20)
        )
        rows = session.execute(stmt).all()
        
        from services.match.service import MatchService
        match_service = MatchService(session)
        import json
        
        for row in rows:
            job = row[0]
            score = row[1]
            print(f"[{score:.4f}] {job.title} (ID: {job.id})")
            
            # Only explain the first 3 to save time/tokens
            if rows.index(row) < 3:
                try:
                    explanation_json = match_service.explain(candidate, job)
                    data = json.loads(explanation_json)
                    confidence = data.get("confidence")
                    print(f"    -> LLM Confidence: {confidence}")
                    print(f"    -> Summary: {data.get('summary')[:100]}...")
                except Exception as e:
                    print(f"    -> LLM Failed: {e}")

if __name__ == "__main__":
    debug_recruiter_matches()
