FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code, corpus and web interface
COPY app/ ./app/
COPY data/ ./data/
COPY static/ ./static/

# The .env file is NOT copied into the image: pass it at run time with --env-file.
# Inside one container the API reaches the MCP server on localhost.
# CACHE_DIR holds the knowledge graph and the vector index: mount a volume on it
# (-v pharma-cache:/app/cache) so the next start takes seconds instead of minutes.
ENV MCP_HOST=127.0.0.1 \
    MCP_PORT=8001 \
    MCP_SERVER_URL=http://127.0.0.1:8001/mcp \
    DATA_DIR=/app/data \
    CACHE_DIR=/app/cache

RUN mkdir -p /app/cache
VOLUME ["/app/cache"]

EXPOSE 8000

# The first start builds the index and graph (several minutes); later starts
# reuse the cache. The healthcheck only checks that the API answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Start the MCP server in the background, then the API in the foreground.
# The API keeps retrying the MCP connection until the server is ready.
CMD ["bash", "-c", "trap 'kill 0' TERM INT; python -m app.mcp_server & uvicorn app.main:app --host 0.0.0.0 --port 8000 & wait -n; exit $?"]