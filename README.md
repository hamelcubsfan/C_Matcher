# Waymo Role Matcher

Match Waymo candidates to open roles using a Gemini-powered retrieval and explanation stack with a recruiter-facing web experience.

## Features

- **FastAPI backend** for job ingestion, candidate uploads, vector retrieval with pgvector, reranking, and Gemini explanations.
- **Next.js 14 web client** styled with Waymo-inspired gradients and logo usage for uploading resumes, inspecting extracted skills, and reviewing match cards.
- **Docker-based local stack** with Postgres 15 + pgvector, plus scripts for one-off ingestion and re-embedding.

## Repository layout

```
apps/
  web/             # Next.js web application
  extension/       # Chrome extension shell (future work)
services/
  api/             # FastAPI gateway
  ingest/          # Boards API polling + normalization + embeddings
  match/           # Retrieval, filters, rerank
  explain/         # Gemini explanations with structured output
  shared/          # Common config, db, embeddings, utils
infra/
  migrations/      # SQL migrations
uploads/           # Local resume storage during development
Dockerfile         # Backend container image
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & docker-compose (optional but recommended)
- Gemini API key and (optionally) Google Cloud credentials if using Vertex
- Postgres 15 with the `vector` extension (pgvector)

Copy `.env.example` to `.env` and provide secrets such as `GEMINI_API_KEY`, `DATABASE_URL`, and `EMBED_DIM`.

## Backend quickstart

```bash
pip install -e .
uvicorn services.api.main:app --reload
```

Then run an ingestion pass and upload a resume:

```bash
python -m services.ingest.run_once
curl -F "file=@/path/to/resume.pdf" http://localhost:8000/candidates/upload
curl http://localhost:8000/match/candidate/<CANDIDATE_ID>?n=5
```

### Docker workflow

```bash
docker compose up --build
```

This starts Postgres with pgvector plus the FastAPI service. Run migrations with:

```bash
docker compose exec db psql -U postgres -d roles -f /migrations/000_init.sql
```

## Web UI quickstart

The web app communicates with the FastAPI gateway via `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000`).

```bash
cd apps/web
npm install
npm run dev
```

Visit `http://localhost:3000` to:

1. Upload a resume (PDF or text).
2. Inspect extracted skills and metadata returned by the backend.
3. Request matches and review retrieval/rerank scores plus Gemini explanations, all styled with Waymo gradients and the Waymo logo sourced from Brandfetch.

For production/static builds:

```bash
npm run export
npx serve out -l 3000
```

`npm run export` writes the static assets to `apps/web/out`, and `npx serve` is an easy way to preview that build locally.

To embed the UI inside the FastAPI container (so `/` serves the Next.js experience), ensure the `apps/web/out` directory exists. The backend auto-detects that directory (configurable via `WEB_STATIC_DIR`) and mounts it at the root path.

## Deployment

- **Backend**: Use the root `Dockerfile` on Render, Fly.io, or similar. The multi-stage build now compiles the Next.js app, so the deployed service exposes both the UI (`/`) and JSON API routes. Configure environment variables mirroring `.env.example` (especially `DATABASE_URL`, `GEMINI_API_KEY`, and `WEB_STATIC_DIR` if you relocate the static export) and point `DATABASE_URL` at a Postgres instance with pgvector enabled.
For production builds:

```bash
npm run build
npm run start
```

## Deployment

- **Backend**: Use the root `Dockerfile` on Render, Fly.io, or similar. Configure environment variables mirroring `.env.example` and point `DATABASE_URL` at a Postgres instance with pgvector enabled.
- **Frontend**: Deploy `apps/web` to Vercel, Netlify, or another Node.js host. Set `NEXT_PUBLIC_API_BASE_URL` to the deployed API.
- **Chrome extension**: `apps/extension` contains the scaffold for an MV3 popup that will reuse the same API endpoints (future enhancement).

## Troubleshooting

- Ensure `EMBED_DIM` matches the vector dimension declared in the Postgres migration.
- If using Gemini reranking, set `RERANK_PROVIDER=gemini`; otherwise the lightweight lexical reranker runs by default.
- For Vertex AI usage, flip `GOOGLE_GENAI_USE_VERTEXAI=True` and supply `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.
- If Render (or another host) serves only JSON at `/`, confirm the Docker build finished the static export or run `npm run export` locally and set `WEB_STATIC_DIR` to the output directory before restarting Uvicorn.

## Roadmap

- Finish Chrome extension UI.
- Add automated ingestion scheduling and health checks.
- Expand evaluation tooling (precision@k, recruiter feedback loops).
