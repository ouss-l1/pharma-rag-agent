# Pharmacology Research Agent (MCP + Hybrid RAG + GraphRAG)

An agentic question-answering system over a corpus of drug notices written in Markdown. It plans the question, retrieves evidence through MCP tools, has the draft reviewed by a critic, and returns a cited answer with a risk flag.

> **Disclaimer:** this project is a technical exercise based only on the provided documents. It is not medical advice. Consult a qualified healthcare professional.

## How it works

```
Question
   |
   v
Planner  -->  ReAct agent  -->  Critic  -->  Finalizer  -->  Answer + sources + risk
              (MCP tools)       (PASS/REVISE)
                  |
                  v
            MCP server (port 8001)
            - Hybrid search (BM25 + Chroma dense vectors)
            - Knowledge graph (NetworkX, triples extracted by the LLM)
            - Section / glossary / drug-list lookups
```

1. **Planner** splits the question into 2 to 5 short steps.
2. **Agent** follows the plan and calls the MCP tools to collect evidence. It must use only what the tools return.
3. **Critic** checks the draft against the evidence and outputs `VERDICT: PASS` or `VERDICT: REVISE`, plus unsupported, ambiguous and missing points.
4. **Finalizer** rewrites the answer using the review, cites the source files, and appends the disclaimer.
5. **Risk** is `high` if the critic asked for a revision, otherwise `low`. Each answer is also sent to an n8n webhook (optional).

### MCP tools

| Tool | Purpose |
|---|---|
| `search_hybrid` | BM25 + dense vector search (default for most questions) |
| `search_bm25_exact` | Exact keyword, dose and number matching |
| `query_knowledge_graph` | Contraindications, interactions and dosage rules from the graph |
| `get_drug_section` | Full section of one drug's document (e.g. `contraindications`) |
| `lookup_glossary` | Medical term definitions from the glossary file |
| `list_available_drugs` | Drugs available in the corpus |

### Retrieval design

- Documents are split with a Markdown-aware splitter (`CHUNK_SIZE = 1000` characters, `CHUNK_OVERLAP = 150`).
- Every chunk is prefixed with `[Drug: name]` so that passages from different drugs are not confused.
- Hybrid search combines BM25 (weight 0.4) and dense embeddings (weight 0.6), with `TOP_K = 5` per retriever.
- The knowledge graph is built once at server start by extracting `(subject, relation, object, effect)` triples from the notices.

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app: /health, /ask, /eval, /
│   ├── agent.py         # Planner -> agent -> critic -> finalizer
│   ├── mcp_server.py    # MCP server exposing the retrieval tools
│   ├── rag.py           # KnowledgeBase: Chroma + BM25 + hybrid retriever
│   ├── graph_rag.py     # Knowledge graph built from the documents
│   ├── eval.py          # Faithfulness / precision / string-similarity evaluator
│   ├── risk.py          # Risk flag + n8n notification
│   └── config.py        # Settings read from .env
├── data/                # Markdown drug notices (*.md) and a glossary file
├── static/
│   └── index.html       # Minimal web interface
├── Dockerfile
├── requirements.txt
└── .env                 # Your local configuration (not committed)
```

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible endpoint for chat and embeddings
- Markdown files in `data/` (drug notices named like `dailymed-warfarin.md`, plus an optional `glossary*.md`)

## Configuration

Create a `.env` file at the project root:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=            # leave empty for the default OpenAI endpoint
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

DATA_DIR=./data
SPLIT_METHOD=markdown       # markdown | token | recursive

# Optional: n8n webhook notified after each answer
N8N_WEBHOOK_URL=

# MCP tool server (these defaults also work inside one container)
MCP_HOST=127.0.0.1
MCP_PORT=8001
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
```

Never commit `.env`. Add it to `.gitignore`.

## Run locally

Install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The system has **two processes**. Start them in this order, in two terminals.

**Terminal 1: MCP server**

```bash
python -m app.mcp_server
```

At startup it embeds the corpus and builds the graph, which can take a minute or more. Wait until you see:

```
MCP server ready: N chunks & M graph edges indexed
Uvicorn running on http://127.0.0.1:8001
```

**Terminal 2: API**

```bash
uvicorn app.main:app --port 8000
```

You should see `Agent ready with 6 MCP tools`. The API keeps retrying the MCP server for up to 3 minutes, so starting it too early is harmless.

Open <http://127.0.0.1:8000> for the web interface.

## API

### `GET /health`

```json
{ "status": "ok", "agent_ready": true }
```

`agent_ready` stays `false` until the MCP tools are loaded.

### `POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the contraindications of warfarin?"}'
```

Response:

```json
{
  "answer": "... (dailymed-warfarin.md)\n\nDisclaimer: ...",
  "sources": ["dailymed-warfarin.md"],
  "plan": "1. ...",
  "review": "VERDICT: PASS ...",
  "risk": "low"
}
```

Returns `503` while the agent is still starting.

### `POST /eval`

Evaluates an answer against its context (faithfulness, context precision, and string similarity if a reference is given):

```json
{
  "question": "...",
  "answer": "...",
  "context": "...",
  "reference_answer": "optional"
}
```

## Docker

Build and run with your `.env`:

```bash
docker build -t pharma-agent .
docker run --env-file .env -p 8000:8000 pharma-agent
```

The container must start **both** processes: the MCP server first, then the API. Inside one container, keep `MCP_HOST=127.0.0.1` and start uvicorn with `--host 0.0.0.0` so `/health` is reachable from outside.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Waiting for MCP server (n/60)` keeps repeating | The MCP server is not running, crashed at startup, or is still building the index. Run it alone and read its output. The log now prints the real cause instead of the generic "TaskGroup" message. |
| `Triple extraction failed ...` in the MCP logs | The chat model or API key is invalid for your endpoint, so the graph ends up empty. Check `OPENAI_MODEL` and `OPENAI_API_KEY`. |
| `ClosedResourceError` / `standalone SSE writer` in the MCP logs | Harmless noise from the MCP SDK when a session closes. It is filtered in `mcp_server.py`. |
| Changes to chunking or data have no effect | The index is rebuilt only when the MCP server starts. Restart it. |
| `/ask` returns `503` | The agent is not ready yet. Check `/health` and the MCP server logs. |
| Answers miss information | Check `sources` and `review` in the response. Increase `TOP_K` or `CHUNK_SIZE` in `config.py` and restart the MCP server. |

## Notes

- Answers are grounded only in the documents in `data/`. If the documents do not contain the answer, the agent is instructed to say so.
- The evidence passed to the critic and finalizer is capped by `MAX_EVIDENCE_CHARS` in `config.py`.