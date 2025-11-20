-- Drop existing indexes that depend on the column type
DROP INDEX IF EXISTS idx_jobs_embedding_ivfflat;
DROP INDEX IF EXISTS idx_candidates_embedding_ivfflat;

-- Alter the columns to the new dimension
ALTER TABLE jobs ALTER COLUMN embedding TYPE VECTOR(768);
ALTER TABLE candidates ALTER COLUMN embedding TYPE VECTOR(768);

-- Recreate the indexes using HNSW (supports > 2000 dims)
CREATE INDEX IF NOT EXISTS idx_jobs_embedding_hnsw ON jobs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_candidates_embedding_hnsw ON candidates USING hnsw (embedding vector_cosine_ops);
