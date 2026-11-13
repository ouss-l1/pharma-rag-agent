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
ENV MCP_HOST=127.0.0.1 \
    MCP_PORT=8001 \
    MCP_SERVER_URL=http://127.0.0.1:8001/mcp \
    DATA_DIR=/app/data

EXPOSE 8000

# The MCP server builds the index and graph at startup (can take a minute or more),
# so give the healthcheck a long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Start the MCP server in the background, then the API in the foreground.
# The API retries the MCP connection for up to 3 minutes, so the order is safe.
CMD ["sh", "-c", "python -m app.mcp_server & exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]