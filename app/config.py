import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
STATIC_DIR = BASE_DIR / "static"
# Cache disque (graphe + index Chroma) : monter un volume Docker dessus
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "cache"))

# LLM & Embedding Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_url = (os.getenv("OPENAI_BASE_URL") or "").strip().strip('"').strip("'")
if _url and not _url.startswith(("http://", "https://")):
    _url = "https://" + _url
    
OPENAI_BASE_URL = _url or None
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# MCP server (tools) settings
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_HOST}:{MCP_PORT}/mcp")

# Markdown RAG chunking settings (sizes are in CHARACTERS, not tokens)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
SPLIT_METHOD = os.getenv("SPLIT_METHOD", "markdown")  # Options: 'markdown', 'token', 'recursive'

# Hybrid Search RRF Weights (BM25 Sparse + Chroma Dense)
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6

# Knowledge graph build: number of parallel LLM calls 
GRAPH_CONCURRENCY = int(os.getenv("GRAPH_CONCURRENCY", "4"))

# Agent Execution Limits
RECURSION_LIMIT = 30
MAX_EVIDENCE_CHARS = 40000

DISCLAIMER = (
    "Disclaimer: this answer is a technical exercise based only on the provided documents. "
    "It is not medical advice. Consult a qualified healthcare professional."
)