FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

COPY server.py ./

EXPOSE 8771

ENV HOST=0.0.0.0
ENV PORT=8771

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8771"]
