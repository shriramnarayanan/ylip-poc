FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

COPY main.py .

EXPOSE 8884
CMD ["uv", "run", "python", "main.py"]
