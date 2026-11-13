FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Uncomment if a dependency needs to compile (e.g. chromadb wheels missing)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY data ./data

RUN useradd --create-home appuser
USER appuser

# Only 8000 is public. The MCP server listens on 127.0.0.1:8001 inside the container.
EXPOSE 8000

# start-period is generous: the MCP server builds embeddings + graph at startup
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# MCP server in the background, API as the main process
CMD ["sh", "-c", "python -m app.mcp_server & exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]