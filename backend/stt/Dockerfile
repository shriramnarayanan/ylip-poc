FROM python:3.12-slim

# ffmpeg for audio format support
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

COPY main.py .

EXPOSE 8881
CMD ["uv", "run", "python", "main.py"]
