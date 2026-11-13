import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
STATIC_DIR = BASE_DIR / "static"

# LLM & Embedding Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# Pure Markdown RAG Chunking Settings (200 tokens chunk / 40 overlap for high medical density)
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
TOP_K = 3
SPLIT_METHOD = os.getenv("SPLIT_METHOD", "markdown")  # Options: 'markdown', 'token', 'recursive'

# Hybrid Search RRF Weights (BM25 Sparse + Chroma Dense)
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6

# Agent Execution Limits
RECURSION_LIMIT = 30
MAX_EVIDENCE_CHARS = 20000

DISCLAIMER = (
    "Disclaimer: this answer is a technical exercise based only on the provided documents. "
    "It is not medical advice. Consult a qualified healthcare professional."
)