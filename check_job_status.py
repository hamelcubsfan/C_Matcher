import sys
import os
from services.shared.db import get_session_factory
from services.shared.models import Job

# Ensure we can import services
sys.path.append(os.getcwd())

def check_job_status():
    with get_session_factory()() as session:
        job_ids = [27, 132, 75]
        jobs = session.query(Job).filter(Job.id.in_(job_ids)).all()
        for job in jobs:
            print(f"Job {job.id} ({job.title}): Status = '{job.posting_status}'")

if __name__ == "__main__":
    check_job_status()
