# Waymo Role Matcher • Web UI

A Next.js 14 recruiter experience that pairs Waymo-inspired theming with the FastAPI backend.

## Getting started

```bash
cd apps/web
npm install
npm run dev
```

The dev server runs on `http://localhost:3000`. Configure the API location via `.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-api-host
```

## User flow

1. **Upload** – Select a resume (PDF or plaintext) and POST to `/candidates/upload`.
2. **Profile review** – Display extracted skills and metadata from the backend response.
3. **Match gallery** – Request `/match/candidate/{id}` to show retrieval scores, rerank scores, and Gemini explanations.
4. **Routing** – Copy structured notes or open the job posting.

## Styling

- Uses the Waymo logomark provided via Brandfetch CDN (configure `next.config.mjs` image domains).
- Applies gradients and accent colors drawn from Waymo’s palette across hero, buttons, and cards.
- Global styles live in `app/globals.css`; component-level tweaks reside alongside the React components in `app/page.tsx`.

## Production build

```bash
npm run export
npx serve out -l 3000
```

`npm run export` writes static assets to `apps/web/out`, and `npx serve` is a convenient way to preview the exported site locally.

Deploy to Vercel/Netlify/etc. with the same `NEXT_PUBLIC_API_BASE_URL` environment variable pointed at your FastAPI deployment.

### Static export for the Python API container

The backend bundles this UI automatically during Docker builds. To preview the exact output locally, run `npm run export` and point `WEB_STATIC_DIR` at `apps/web/out` when launching FastAPI so `/` serves the Waymo-themed interface while API endpoints remain available under the same origin.
npm run build
npm run start
```

Deploy to Vercel/Netlify/etc. with the same `NEXT_PUBLIC_API_BASE_URL` environment variable pointed at your FastAPI deployment.
