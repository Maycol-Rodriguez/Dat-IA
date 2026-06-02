FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV SEED=42

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache

COPY . .

CMD ["uv", "run", "src/main.py"]
