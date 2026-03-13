FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --all-extras

COPY src/ ./src/

EXPOSE 7860

# We use the internal uv run command to automatically wrap the execution inside the python venv shell
CMD ["uv", "run", "python", "src/main.py"]
