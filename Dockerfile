FROM node:20 AS web-builder
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./apps/web/
WORKDIR /web/apps/web
RUN npm ci
COPY apps/web/ /web/apps/web
ARG NEXT_PUBLIC_API_BASE_URL=""
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
RUN npm run export

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . .
COPY --from=web-builder /web/apps/web/out ./apps/web/out
RUN pip install --upgrade pip \
    && pip install -e .

CMD ["sh", "-c", "python -m services.shared.migrate && uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
