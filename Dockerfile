# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93
ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --frozen \
        --no-dev \
        --no-install-project

FROM ${PYTHON_IMAGE} AS runtime

ARG VCS_REF=unknown
ARG VERSION=0.2.0

LABEL org.opencontainers.image.title="Dat-IA API" \
      org.opencontainers.image.description="API de consultas en lenguaje natural sobre PostgreSQL" \
      org.opencontainers.image.source="https://github.com/Maycol-Rodriguez/Dat-IA" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface

RUN groupadd --gid 10001 app \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --create-home \
        --shell /usr/sbin/nologin \
        app \
    && mkdir -p \
        /app/chroma_db \
        /app/.cache/huggingface \
    && chown -R 10001:10001 \
        /app/chroma_db \
        /app/.cache

COPY --from=builder /app/.venv /app/.venv
COPY --chown=10001:10001 app ./app
COPY --chown=10001:10001 data ./data

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
