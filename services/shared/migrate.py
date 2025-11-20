import logging
import sys
from pathlib import Path
from sqlalchemy import text
from services.shared.db import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy.exc import OperationalError

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(OperationalError)
)
def run_migrations():
    """Reads all SQL files in infra/migrations and applies them."""
    logger.info("Starting database migration...")
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    migrations_dir = root_dir / "infra" / "migrations"
    
    if not migrations_dir.exists():
        logger.error(f"Migrations directory not found at {migrations_dir}")
        sys.exit(1)
        
    # Get all .sql files and sort them
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    if not migration_files:
        logger.warning("No migration files found.")
        return

    engine = get_engine()
    with engine.connect() as connection:
        with connection.begin():
            for migration_file in migration_files:
                logger.info(f"Applying migration: {migration_file.name}")
                with open(migration_file, "r") as f:
                    sql_content = f.read()
                
                try:
                    connection.execute(text(sql_content))
                except Exception as e:
                    logger.error(f"Failed to apply {migration_file.name}: {e}")
                    raise
            
    logger.info("All migrations completed successfully.")

if __name__ == "__main__":
    run_migrations()
