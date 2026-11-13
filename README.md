# 💊 Pharmacology AI Agent (Hybrid RAG + Dynamic GraphRAG + Slack & n8n)

An enterprise-grade, evidence-backed Pharmacology AI Research Assistant built with **FastAPI**, **LangGraph**, **ChromaDB**, **NetworkX**, **Slack**, and **n8n**.

The agent processes raw Markdown (`.md`) medical documentation and drug monographs to answer clinical queries with explicit source citations. It features a multi-agent guardrail system (Planner, ReAct Execution, Critic Audit, and Finalizer) and streams high-risk safety alerts directly to dedicated Slack channels.

## 📐 Architecture Overview

The system supports a **Dual-Entry Architecture**, meaning users can ask questions directly through the Web UI or natively within Slack. Regardless of where the question originates, the system returns the formulated answer to the chat immediately, while simultaneously evaluating the Critic's safety review to send asynchronous risk alerts.

```text
  [ Web UI (localhost:8000) ]                 [ Slack (#ask-pharma-agent) ]
              │                                             │
              │ 1a. Ask via Web Chat                        │ 1b. Ask via Slack
              │                                             ▼
              │                             ┌───────────────────────────────┐
              │                             │    Cloudflare / Ngrok Tunnel  │
              │                             └───────────────┬───────────────┘
              │                                             │ 2. Webhook Event
              │                                             ▼
              │                             ┌───────────────────────────────┐
              │                             │    n8n Automation Engine      │
              │                             └───────────────┬───────────────┘
              │                                             │ 3. HTTP POST /ask
              ▼                                             ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                    FastAPI Agent Engine                       │
        │             (Hybrid RAG + Dynamic Graph + Multi-Agent)        │
        └───────┬───────────────────────────────────────────────┬───────┘
                │                                               │
                │ 4a. Formatted Answer                          │ 4b. Async Risk Alert
                │    (Delivered instantly)                      │    (Triggered by Critic)
                ▼                                               ▼
     [ Returns to Web UI ]                             [ n8n Risk Webhook ]
              OR                                                │
   [ Returns to Slack via n8n ]                                 ▼
                                                 [ Slack (#pharma-agent-risk1) ]
```

## ✨ Key Features

* **Markdown-Native RAG Pipeline**: Engineered specifically for high-density medical Markdown (`.md`) monographs.

* **Hybrid Search Engine**: Combines **BM25** keyword retrieval and **ChromaDB** dense vector embeddings via Reciprocal Rank Fusion (RRF).

* **Dynamic LLM GraphRAG**: Automatically extracts clinical entities and relationship triples (`CONTRAINDICATED_WITH`, `INTERACTS_WITH`, `DOSAGE_RULE`) into an in-memory **NetworkX** graph with JSON disk caching.

* **Multi-Agent Governance & Safety**:
  * **Planner**: Decomposes clinical questions into structured execution steps.
  * **ReAct Agent**: Executes hybrid retrieval and graph queries.
  * **Critic**: Audits draft answers against raw evidence for unsupported or ambiguous claims.
  * **Finalizer**: Synthesizes verified answers with inline parenthetical citations e.g. `(dailymed-warfarin.md)`.

* **Event-Driven Slack & n8n Integration**: Bidirectional webhook automation for channel responses and asynchronous risk alerts.

* **Evaluation Endpoint (`/eval`)**: Evaluates Context Precision, Faithfulness, and Levenshtein String Similarity.

## 📂 Repository Structure

```text
.
├── app/
│   ├── agent.py          # Multi-agent LangGraph workflow (Planner, ReAct, Critic, Finalizer)
│   ├── config.py         # System configuration & environment variables
│   ├── eval.py           # RAG Evaluation module (Faithfulness, Precision, String Similarity)
│   ├── graph_rag.py      # Dynamic NetworkX GraphRAG engine & triple extraction
│   ├── main.py           # FastAPI application endpoints (/ask, /eval, /health)
│   ├── rag.py            # BM25 + Chroma Hybrid Retriever pipeline
│   └── risk.py           # Risk assessment & background n8n webhook notification dispatcher
├── data/                 # Raw Markdown corpus (.md drug monographs & glossary)
│   ├── dailymed-acetaminophen.md
│   ├── dailymed-atorvastatin.md
│   ├── dailymed-ibuprofen.md
│   ├── dailymed-sertraline.md
│   ├── dailymed-warfarin.md
│   └── glossary.md
├── n8n_workflows/        # Exported n8n workflow JSON templates
│   ├── slack_pharma_qa_workflow.json
│   └── pharma_risk_alerts_workflow.json
├── static/
│   └── index.html        # Clean light-theme web UI
├── docker-compose.yml    # Multi-container orchestration (FastAPI + n8n)
├── Dockerfile            # Container build recipe for FastAPI app
├── requirements.txt      # Python dependencies (Pinned for Python 3.11/3.12)
└── README.md
```

## ⚙️ Prerequisites

* **Python 3.11 or 3.12** *(Python 3.14 is currently incompatible with Pydantic type evaluation)*
* **Docker & Docker Compose** (Optional, for containerized deployment)
* **Cloudflare CLI (`cloudflared`)** or **Ngrok** (For exposing local n8n webhooks to Slack)
* **Slack Workspace** with administrative access to create Slack Apps

## 🔐 Environment Variables Configuration

Create a `.env` file in the root directory:

```env
# LLM & OpenAI-compatible Endpoint Settings
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://llm.ai.syntax-rnd.com
OPENAI_MODEL=gpt-5.6-terra
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# n8n Integration & Webhook Target
N8N_WEBHOOK_URL=https://guru-joel-greater-mine.trycloudflare.com/webhook/pharma-risk

# Directory Paths & Chunking
DATA_DIR=./data
SPLIT_METHOD=markdown
```

> **Note on Webhook URLs**: Quick tunnels (`cloudflared tunnel --url http://localhost:5678`) generate a new random domain on every restart. If your Cloudflare URL changes, update `N8N_WEBHOOK_URL` in `.env` accordingly, or use Ngrok with a permanent free static domain.

## 🚀 Quickstart Guide

### Option 1: Running Locally (Development Mode)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pharma-agent.git
   cd pharma-agent
   ```

2. **Create & Activate Virtual Environment**:
   ```powershell
   # Windows PowerShell (Python 3.11 or 3.12)
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the FastAPI Backend**:
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the Application**:
   * Web UI: `http://localhost:8000`
   * Swagger API Docs: `http://localhost:8000/docs`

### Option 2: Containerized Deployment with Docker Compose

Run both the FastAPI backend and n8n engine using a single command:

```bash
docker compose up -d --build
```

* **FastAPI Backend**: Runs on `http://localhost:8000`
* **n8n Workflow Engine**: Runs on `http://localhost:5678`

## 🔗 Slack, Cloudflare & n8n Integration Setup

### Step 1: Start Cloudflare Tunnel

Generate a public HTTPS tunnel to expose your local n8n instance (`port 5678`):

```bash
cloudflared tunnel --url http://localhost:5678
```
*Copy the generated tunnel URL (e.g., `https://guru-joel-greater-mine.trycloudflare.com`).*

### Step 2: Import n8n Workflows

1. Open n8n in your browser at `http://localhost:5678`.
2. Click **Workflows** -> **Import from File**.
3. Import the JSON files located in `./n8n_workflows/`:
   * `slack_pharma_qa_workflow.json` (Main Slack listener and query dispatcher)
   * `pharma_risk_alerts_workflow.json` (Background Critic risk log publisher)
4. Update the **HTTP Request** node inside n8n:
   * If n8n runs inside Docker, set the URL to: `http://host.docker.internal:8000/ask`
   * If running locally without Docker, set to: `http://localhost:8000/ask`
5. Activate both workflows in n8n.

### Step 3: Configure Slack App

1. Go to [Slack API Apps](https://api.slack.com/apps) and create a new App in your workspace.
2. Under **OAuth & Permissions**, add the following Bot Token Scopes:
   * `chat:write`
   * `channels:history`
   * `app_mentions:read`
3. Install the App to your Slack Workspace.
4. Go to **Event Subscriptions**:
   * Enable Events.
   * Set **Request URL** to your Cloudflare URL: `https://guru-joel-greater-mine.trycloudflare.com/webhook/slack-event`
   * Subscribe to Bot Events: `app_mention` and `message.channels`.
5. Invite the bot to your channels:
   * `#ask-pharma-agent`
   * `#pharma-agent-risk1`

## 🌐 API Reference

### `POST /ask`
Submits a clinical question to the multi-agent pipeline.

* **Request Body**:
  ```json
  {
    "question": "What are the main contraindications of Warfarin?"
  }
  ```
* **Response (`200 OK`)**:
  ```json
  {
    "answer": "Warfarin is contraindicated in:\n- **Pregnancy**... (dailymed-warfarin.md)\n\nDisclaimer: ...",
    "sources": ["dailymed-warfarin.md"],
    "plan": "1. Search contraindications for Warfarin.\n2. Summarize findings.",
    "review": "VERDICT: PASS\nUNSUPPORTED CLAIMS: none...",
    "risk": "low"
  }
  ```

### `POST /eval`
Runs automated RAG evaluation metrics on a query-context-answer triplet.

* **Request Body**:
  ```json
  {
    "question": "What is the interaction between Sertraline and Warfarin?",
    "context": "Concomitant use of sertraline and warfarin may potentiate bleeding risk...",
    "answer": "Combining Sertraline and Warfarin increases risk of bleeding."
  }
  ```

## 📄 License

This project is open-source and available under the **MIT License**.