CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    id BIGSERIAL PRIMARY KEY,
    greenhouse_job_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    team TEXT,
    location TEXT,
    must_have_skills JSONB DEFAULT '[]',
    nice_to_have_skills JSONB DEFAULT '[]',
    description TEXT,
    absolute_url TEXT,
    posting_status TEXT NOT NULL DEFAULT 'open',
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(posting_status);
CREATE INDEX IF NOT EXISTS idx_jobs_embedding_hnsw ON jobs USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name TEXT,
    email TEXT UNIQUE,
    location TEXT,
    resume_url TEXT,
    parsed_profile JSONB DEFAULT '{}'::jsonb,
    skills JSONB DEFAULT '[]',
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_candidates_embedding_hnsw ON candidates USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS matches (
    id BIGSERIAL PRIMARY KEY,
    candidate_id UUID NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    job_id BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    retrieval_score DOUBLE PRECISION,
    rerank_score DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    reason_codes JSONB DEFAULT '[]',
    explanation TEXT,
    routed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_candidate_job ON matches(candidate_id, job_id);
