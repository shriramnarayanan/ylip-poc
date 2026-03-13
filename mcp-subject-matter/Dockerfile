FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

COPY retrieval.py server.py ./
COPY subjects/ ./subjects/

# Expose port exactly as requested by user
EXPOSE 8770

# Run mcp server in SSE mode bounded to all interfaces
ENV HOST=0.0.0.0
ENV PORT=8770
ENV TRANSPORT=sse

CMD ["uv", "run", "python", "server.py"]
