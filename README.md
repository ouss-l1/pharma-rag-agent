# 💊 Pharmacology AI Agent

**Hybrid RAG + Dynamic GraphRAG + Multi-Agent Guardrails + Slack & n8n automation**

An evidence-backed pharmacology research assistant. It reads drug monographs written in Markdown (`.md`), answers clinical questions with **explicit source citations** (e.g. `(dailymed-warfarin.md)`), audits its own answers with a "Critic" agent, and pushes **risk alerts** to a dedicated Slack channel when the Critic is not satisfied.

> ⚠️ **Educational project.** Answers are generated only from the provided documents and are **not medical advice**.

---

## Table of Contents

1. [Key Features](#1-key-features)
2. [Architecture](#2-architecture)
3. [Tools & Technologies Explained](#3-tools--technologies-explained)
4. [Repository Structure](#4-repository-structure)
5. [How the Agent Works](#5-how-the-agent-works)
6. [Prerequisites](#6-prerequisites)
7. [Configuration (`.env`)](#7-configuration-env)
8. [Running the FastAPI Backend](#8-running-the-fastapi-backend)
9. [Running n8n with Docker (command reference)](#9-running-n8n-with-docker-command-reference)
10. [Exposing n8n with a Tunnel (Cloudflare / Ngrok)](#10-exposing-n8n-with-a-tunnel-cloudflare--ngrok)
11. [Creating the Slack App (step by step)](#11-creating-the-slack-app-step-by-step)
12. [Slack Credentials & OAuth Explained](#12-slack-credentials--oauth-explained)
13. [Building the n8n Workflows](#13-building-the-n8n-workflows)
14. [Recommended Startup Order](#14-recommended-startup-order)
15. [API Reference](#15-api-reference)
16. [Testing](#16-testing)
17. [Command Cheat Sheet](#17-command-cheat-sheet)
18. [Troubleshooting](#18-troubleshooting)
19. [Security Checklist](#19-security-checklist)
20. [Known Limitations](#20-known-limitations)

---

## 1. Key Features

| Feature | Description |
|---|---|
| **Markdown-native RAG** | Built for dense medical Markdown monographs; chunking is Markdown-aware. |
| **Hybrid search** | BM25 (keyword) + ChromaDB (dense embeddings) merged by weighted Reciprocal Rank Fusion. |
| **Dynamic GraphRAG** | An LLM extracts clinical triples (`CONTRAINDICATED_WITH`, `INTERACTS_WITH`, `CAUSES_SIDE_EFFECT`, `DOSAGE_RULE`) into an in-memory NetworkX graph at startup. |
| **Multi-agent governance** | Planner → ReAct agent → Critic → Finalizer. |
| **Citations** | Every claim must cite its source file in parentheses. |
| **Risk alerts** | If the Critic returns `VERDICT: REVISE`, the request is flagged `high` risk and an alert is posted to Slack through n8n. |
| **Dual entry** | Ask from the Web UI or directly from Slack. |
| **Evaluation endpoint** | `/eval` scores faithfulness, context precision and string similarity. |

---

## 2. Architecture

The system has **two entry points** (Web UI and Slack) that both end up in the same FastAPI engine. The answer is returned immediately, while the Critic's verdict is evaluated in a **background task** that may trigger an asynchronous risk alert.

```
  [ Web UI  http://localhost:8000 ]            [ Slack  #ask-pharma-agent ]
               │                                          │
               │ 1a. Question in browser                  │ 1b. Question / @mention
               │                                          ▼
               │                              ┌───────────────────────────┐
               │                              │  Public HTTPS tunnel      │
               │                              │  (Cloudflare / Ngrok)     │
               │                              └─────────────┬─────────────┘
               │                                            │ 2. Slack Event (webhook)
               │                                            ▼
               │                              ┌───────────────────────────┐
               │                              │  n8n  (Docker, :5678)     │
               │                              │  Workflow 1: Slack Q&A    │
               │                              └─────────────┬─────────────┘
               │                                            │ 3. HTTP POST /ask
               ▼                                            ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                    FastAPI Agent Engine (:8000)                 │
       │   Planner → ReAct Agent → Critic → Finalizer                    │
       │   Tools: Hybrid RAG (BM25 + Chroma) · Knowledge Graph · Glossary│
       └───────┬─────────────────────────────────────────────┬───────────┘
               │ 4a. JSON answer (immediate)                 │ 4b. Background task
               ▼                                             ▼
   [ Back to Web UI ]  or  [ Back to n8n → Slack ]   POST N8N_WEBHOOK_URL (/webhook/pharma-risk)
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │ n8n Workflow 2: Risk Alert│
                                                │ Webhook → If → OpenAI →   │
                                                │ Slack message             │
                                                └─────────────┬─────────────┘
                                                              ▼
                                                [ Slack  #pharma-agent-risk1 ]
```

### Component responsibilities

| Component | Role |
|---|---|
| **FastAPI** | HTTP API (`/ask`, `/eval`, `/health`, `/`) and orchestration of all Python modules. |
| **Knowledge Base (`rag.py`)** | Loads `.md` files, chunks them, builds BM25 + Chroma indexes, exposes search/section/glossary helpers. |
| **GraphRAG engine (`graph_rag.py`)** | Builds and queries the NetworkX knowledge graph. |
| **Agent (`agent.py`)** | LangChain/LangGraph multi-step reasoning with tools, review and finalization. |
| **Risk module (`risk.py`)** | Converts the Critic's verdict into `low`/`high` and sends the event to n8n. |
| **n8n Workflow 1** | Slack message → `POST /ask` → reply in Slack. |
| **n8n Workflow 2** | Receives risk events → filters high risk → formats an alert with an LLM → posts to Slack. |
| **Tunnel** | Gives Slack a public HTTPS URL that reaches your local n8n. |
| **Slack App** | Identity, permissions and event subscriptions for the bot. |

---

## 3. Tools & Technologies Explained

### Backend (Python)

| Tool | Why it is used here |
|---|---|
| **Python 3.11 / 3.12** | Runtime. Python 3.14 is currently incompatible with Pydantic's type evaluation in this stack. |
| **FastAPI** | Web framework exposing the REST endpoints; also generates Swagger docs at `/docs`. |
| **Uvicorn** | ASGI server that runs FastAPI. `--reload` restarts it on code changes (dev only). |
| **Pydantic** | Validates request/response bodies (`AskRequest`, `AskResponse`, `EvalRequest`) and defines structured LLM output (`KnowledgeTriple`, `DocumentTriples`). |
| **python-dotenv** | Loads the `.env` file into environment variables (`config.py`). |
| **LangChain** (`langchain`, `langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`, `langchain-chroma`, `langchain-classic`) | Prompt templates, LLM wrappers, document loaders, text splitters, retrievers, tools and agent creation. |
| **LangGraph** | Execution engine behind the ReAct-style agent (tool-calling loop with a recursion limit). |
| **OpenAI-compatible API** | Chat model + embedding model. Any provider exposing the OpenAI protocol works through `OPENAI_BASE_URL`. |
| **ChromaDB** | Vector store for dense (semantic) retrieval. Currently in-memory and rebuilt at each startup. |
| **rank_bm25 (BM25Retriever)** | Sparse keyword retrieval, ideal for exact drug names and dosages. |
| **EnsembleRetriever** | Combines BM25 + dense results with weights (0.4 / 0.6) using Reciprocal Rank Fusion. |
| **NetworkX** | In-memory directed graph storing drug/condition relationships. |
| **httpx** | Sends the risk webhook to n8n (5 s timeout, never raises). |
| **difflib** | Provides the string similarity ratio in `/eval`. |

### Automation & infrastructure

| Tool | Why it is used here |
|---|---|
| **n8n** | Low-code automation engine. Receives Slack events, calls the agent, and publishes risk alerts. |
| **Docker** | Runs n8n in an isolated container with a persistent data volume. |
| **Cloudflare Tunnel (`cloudflared`)** | Creates a public HTTPS URL pointing at `localhost:5678` so Slack can reach n8n. |
| **Ngrok** *(alternative)* | Same purpose; the free plan offers a stable static domain. |
| **Slack App** | Bot user, scopes, event subscriptions. |

---

## 4. Repository Structure

```
.
├── app/
│   ├── __init__.py
│   ├── agent.py          # Planner, ReAct agent (6 tools), Critic, Finalizer
│   ├── config.py         # Environment variables & tuning constants
│   ├── eval.py           # RAGEvaluator: faithfulness, context precision, string similarity
│   ├── graph_rag.py      # NetworkX GraphRAG: triple extraction & graph queries
│   ├── main.py           # FastAPI app: /ask, /eval, /health, /
│   ├── rag.py            # KnowledgeBase: loading, chunking, BM25 + Chroma + hybrid
│   └── risk.py           # assess_risk() and notify_n8n()
├── data/                 # Markdown corpus
│   ├── dailymed-<drug>.md
│   └── glossary.md
├── n8n_workflows/        # Exported n8n workflows (JSON)
│   ├── slack_pharma_qa_workflow.json
│   └── pharma_risk_alerts_workflow.json
├── static/
│   └── index.html        # Web UI
├── docker-compose.yml    # Optional: multi-container setup
├── Dockerfile            # Optional: container image for the API
├── requirements.txt
├── .env                  # Secrets (never commit)
└── README.md
```

**Corpus naming rule:** files must be named `dailymed-<drug>.md` (the drug name is whatever follows the first `-`). `glossary.md` is treated as the glossary, not as a drug.

---

## 5. How the Agent Works

### 5.1 Pipeline

```
Question
   │
   ▼
┌──────────┐   numbered plan (2–5 steps)
│ Planner  │───────────────────────────────┐
└──────────┘                               ▼
                                   ┌────────────────┐   tool calls   ┌───────────────┐
                                   │  ReAct Agent   │◄──────────────►│ 6 retrieval   │
                                   │ (draft answer) │                │ tools         │
                                   └───────┬────────┘                └───────────────┘
                                           │ draft + collected evidence
                                           ▼
                                   ┌────────────────┐
                                   │     Critic     │ VERDICT / UNSUPPORTED / AMBIGUOUS / MISSING
                                   └───────┬────────┘
                                           ▼
                                   ┌────────────────┐
                                   │   Finalizer    │ fixes the draft, adds citations + disclaimer
                                   └───────┬────────┘
                                           ▼
                                  answer + sources + plan + review + risk
```

1. **Planner**: breaks the question into 2–5 short steps naming the drug(s).
2. **ReAct agent**: follows the plan and calls tools. It is instructed to use **only** tool output, never outside knowledge, and to retrieve each drug separately for comparisons.
3. **Evidence collection**: all tool outputs are concatenated and truncated to `MAX_EVIDENCE_CHARS` (20 000).
4. **Sources**: extracted from the evidence with the pattern `[file.md]`.
5. **Critic**: compares the draft with the evidence and returns a structured review.
6. **Finalizer**: rewrites the draft (removes unsupported claims, adds missing points, flags ambiguity) and appends the disclaimer.
7. **Risk**: `high` if the review contains `VERDICT: REVISE`, otherwise `low`.

### 5.2 Agent tools

| Tool | Purpose |
|---|---|
| `search_hybrid` | BM25 + dense search. Default tool for contraindications, dosages, general questions. |
| `search_bm25_exact` | Keyword-only search for exact terms and dosage strings. |
| `query_knowledge_graph` | Returns graph edges for a drug/condition entity (incoming and outgoing). |
| `get_drug_section` | Returns a whole Markdown section (matched by heading) for one drug, max 4000 chars per file. |
| `lookup_glossary` | Finds lines in `glossary.md` that contain a term. |
| `list_available_drugs` | Lists drugs and files available in the corpus. |

### 5.3 GraphRAG

At startup, each raw Markdown document that contains one of the keywords *contraindication, interaction, warning, adverse* is sent to the LLM with structured output (`DocumentTriples`). Each triple becomes a directed edge:

```
(Warfarin) -[CONTRAINDICATED_WITH]-> (Pregnancy) {effect, source file}
```

`query_graph_relations(entity)` returns outgoing edges for the entity and incoming edges pointing to it.

### 5.4 Tuning constants (`app/config.py`)

| Constant | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 200 / 40 | Chunking for dense medical text. |
| `TOP_K` | 3 | Passages returned per retriever. |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | 0.4 / 0.6 | Fusion weights. |
| `SPLIT_METHOD` | `markdown` | `markdown`, `token` or `recursive`. |
| `RECURSION_LIMIT` | 30 | Max agent steps. |
| `MAX_EVIDENCE_CHARS` | 20000 | Evidence sent to Critic/Finalizer. |

---

## 6. Prerequisites

| Requirement | Notes |
|---|---|
| Python **3.11 or 3.12** | Not 3.14. |
| Git | To clone the repo. |
| Docker Desktop | To run n8n. |
| `cloudflared` (or Ngrok) | Public tunnel for Slack. |
| Slack workspace | You need permission to create/install apps. |
| LLM API key | OpenAI or any OpenAI-compatible provider. |
| An OpenAI credential for n8n | Used by the "Message a model" node of the risk workflow. |

Install on Windows (PowerShell):

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
winget install -e --id Docker.DockerDesktop
winget install -e --id Cloudflare.cloudflared
# Alternative tunnel
# winget install -e --id Ngrok.Ngrok
```

Allow PowerShell to activate virtual environments:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Verify:

```powershell
py -3.11 --version
git --version
docker --version
cloudflared --version
```

---

## 7. Configuration (`.env`)

Create `.env` in the project root:

```env
# LLM & embeddings (OpenAI-compatible)
OPENAI_API_KEY=<your_api_key>
OPENAI_BASE_URL=<provider_base_url>
OPENAI_MODEL=<chat_model_name>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Where FastAPI sends risk events (n8n Workflow 2, Webhook node, Production URL)
N8N_WEBHOOK_URL=<n8n_risk_webhook_production_url>

# Corpus & chunking
DATA_DIR=./data
SPLIT_METHOD=markdown
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | API key of your LLM provider. |
| `OPENAI_BASE_URL` | ➖ | Leave empty for the official OpenAI API; set it for compatible providers. |
| `OPENAI_MODEL` | ➖ | Chat model (default `gpt-4o-mini`). |
| `OPENAI_EMBEDDING_MODEL` | ➖ | Embedding model (default `text-embedding-3-small`). |
| `N8N_WEBHOOK_URL` | ➖ | If empty, risk alerts are silently disabled. |
| `DATA_DIR` | ➖ | Folder containing the `.md` files. |
| `SPLIT_METHOD` | ➖ | `markdown`, `token`, `recursive`. |

**Choosing `N8N_WEBHOOK_URL`.** FastAPI runs on your machine and n8n's port is published on `localhost:5678`, so the simplest value is the local one:

```
http://localhost:5678/webhook/<risk-webhook-path>
```

A public tunnel URL also works, but it changes on every restart (quick tunnels) and adds an unnecessary network hop.

> Use the **Production URL** (`/webhook/...`) for a published workflow. The **Test URL** (`/webhook-test/...`) only works while the editor is listening for a test event.

---

## 8. Running the FastAPI Backend

```powershell
# 1. Clone
git clone <repository_url>
cd <project_folder>

# 2. Create the virtual environment
py -3.11 -m venv .venv

# 3. Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux:  source .venv/bin/activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Start the API
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Command part | Meaning |
|---|---|
| `py -3.11 -m venv .venv` | Creates an isolated Python environment in `.venv`. |
| `Activate.ps1` | Makes `python`/`pip` use that environment. |
| `pip install -r requirements.txt` | Installs all pinned dependencies. |
| `python -m uvicorn` | Runs Uvicorn through the active interpreter. |
| `app.main:app` | Module `app/main.py`, object `app`. |
| `--reload` | Auto-restart on file changes (development only). |
| `--host 0.0.0.0` | Listen on all interfaces (required so Docker containers can reach it). |
| `--port 8000` | HTTP port. |

On startup the app loads the corpus, creates embeddings, builds the BM25 index, extracts the graph with the LLM, and logs:

```
Markdown Hybrid Agent Ready: N chunks & M graph edges indexed
```

Open:

- **Web UI**: <http://localhost:8000>
- **Swagger docs**: <http://localhost:8000/docs>
- **Health check**: <http://localhost:8000/health>

---

## 9. Running n8n with Docker (command reference)

### 9.1 Start n8n

```bash
docker run -d --name n8n -p 5678:5678 -e WEBHOOK_URL=<PUBLIC_TUNNEL_URL> -e N8N_PROXY_HOPS=1 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

| Part | Meaning |
|---|---|
| `docker run` | Create and start a new container. |
| `-d` | Detached: run in the background. |
| `--name n8n` | Name the container `n8n` (used by `stop`, `start`, `logs`, `rm`). |
| `-p 5678:5678` | Publish container port 5678 on host port 5678 (`host:container`). n8n UI → <http://localhost:5678>. |
| `-e WEBHOOK_URL=<PUBLIC_TUNNEL_URL>` | Public base URL n8n uses to display and register webhook URLs. Must be the tunnel URL **without any path**. |
| `-e N8N_PROXY_HOPS=1` | Tells n8n it sits behind 1 reverse proxy (the tunnel) so it trusts `X-Forwarded-*` headers. Prevents proxy-related header errors. |
| `-v n8n_data:/home/node/.n8n` | Named volume storing workflows, credentials and executions so they survive container removal. |
| `docker.n8n.io/n8nio/n8n` | Official n8n image (latest tag). |

> Windows tip: type the command on **one line**. Multi-line continuation uses `` ` `` in PowerShell, `^` in cmd and `\` in Bash.

### 9.2 Lifecycle commands

```bash
docker ps                  # list running containers
docker logs -f n8n         # follow n8n logs (Ctrl+C to leave)
docker stop n8n            # stop the container (data kept in the volume)
docker start n8n           # start it again with the SAME settings
docker rm n8n              # delete the container (the volume n8n_data is kept)
docker pull docker.n8n.io/n8nio/n8n   # download the latest image
```

### 9.3 When the tunnel URL changes

Environment variables are fixed when a container is **created**. A new `WEBHOOK_URL` therefore requires recreating the container:

```bash
docker stop n8n
docker rm n8n
docker run -d --name n8n -p 5678:5678 -e WEBHOOK_URL=<NEW_PUBLIC_TUNNEL_URL> -e N8N_PROXY_HOPS=1 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

Your workflows and credentials remain, because they live in `n8n_data`. Then update the **Request URL** in Slack (see [Section 11](#11-creating-the-slack-app-step-by-step)).

### 9.4 Reaching FastAPI from inside the container

Inside the n8n container `localhost` means the container itself, not your PC. Use:

| Where FastAPI runs | URL for the HTTP Request node |
|---|---|
| On your host (Docker Desktop, Windows/macOS) | `http://host.docker.internal:8000/ask` |
| On your host (Linux) | add `--add-host=host.docker.internal:host-gateway` to `docker run`, then use the same URL |
| In another container (Compose) | `http://<service_name>:8000/ask` |

---

## 10. Exposing n8n with a Tunnel (Cloudflare / Ngrok)

Slack can only send events to a **public HTTPS** address. A tunnel forwards that address to your local n8n.

### Cloudflare quick tunnel

```bash
cloudflared tunnel --url http://localhost:5678
```

| Part | Meaning |
|---|---|
| `cloudflared tunnel` | Runs a Cloudflare tunnel. |
| `--url http://localhost:5678` | Local service to expose (n8n). |

The terminal prints a URL like `https://<random-words>.trycloudflare.com`. **Keep this terminal open.**

> Quick tunnels get a **new random URL on every restart**. Every time it changes you must: (1) recreate the n8n container with the new `WEBHOOK_URL`, (2) update the Slack Event Subscriptions Request URL, and (3) update `N8N_WEBHOOK_URL` in `.env` if you use the public URL there.

### Ngrok alternative

```bash
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
ngrok http 5678 --domain=<your-static-domain>.ngrok-free.app
```

A static domain removes the "URL changes" problem.

---

## 11. Creating the Slack App (step by step)

### Step 1: Create the app

1. Go to <https://api.slack.com/apps>.
2. Click **Create New App** → **From scratch**.
3. Enter an **App Name** (e.g. `Pharma Agent`) and choose your **workspace**.
4. Click **Create App**.

### Step 2: Name the bot user

1. In the left menu open **App Home**.
2. Under **Your App's Presence in Slack**, set a **Display Name** and **Default Username** (e.g. `pharma-agent`).
3. Optionally enable **Always Show My Bot as Online**.

### Step 3: Add Bot Token Scopes

1. Open **OAuth & Permissions**.
2. Scroll to **Scopes → Bot Token Scopes** and add:

| Scope | Why it is needed |
|---|---|
| `chat:write` | Post answers and alerts. |
| `channels:history` | Read messages in public channels (event `message.channels`). |
| `app_mentions:read` | Receive `@bot` mentions (event `app_mention`). |
| `channels:read` | Recommended: lets n8n list channels in dropdowns. |
| `chat:write.public` | Optional: post in public channels the bot has not joined. |
| `groups:history` | Optional: only for **private** channels (also needs event `message.groups`). |

### Step 4: Install the app

1. Still in **OAuth & Permissions**, click **Install to Workspace** (or **Install App**).
2. Review permissions and click **Allow**.
3. Copy the **Bot User OAuth Token**; it starts with `xoxb-`. This is the credential n8n will use.

> Whenever you add scopes later, Slack shows a banner asking you to **reinstall** the app. Do it, otherwise the new scopes are not active (the token stays `xoxb-…` but gains the new permissions).

### Step 5: Collect the app credentials

Open **Basic Information → App Credentials**:

| Value | Where | Used for |
|---|---|---|
| **Signing Secret** | Basic Information | Verifying that requests really come from Slack. |
| **Client ID / Client Secret** | Basic Information | Only for the OAuth2 flow ([Section 12](#12-slack-credentials--oauth-explained)). |

### Step 6: Create the Slack credential in n8n

1. In n8n (<http://localhost:5678>) go to **Credentials → Create credential**.
2. Search for **Slack API**.
3. Authentication: **Access Token**.
4. Paste the `xoxb-…` token. Paste the **Signing Secret** too if your n8n version shows a *Signature Secret* field.
5. **Save** and run the connection test.

Use this credential in both the **Slack Trigger** and **Send a message** nodes.

### Step 7: Get the n8n Request URL

1. Open the Q&A workflow and click the **Slack Trigger** node.
2. Copy the **Production URL**, which looks like:
   `<PUBLIC_TUNNEL_URL>/webhook/<id>/webhook`
3. The workflow must be **Published/Active** for this URL to respond. For a first test you can use the **Test URL** after clicking *Listen for test event*.

### Step 8: Enable Event Subscriptions

1. In Slack, open **Event Subscriptions** and switch **Enable Events** on.
2. Paste the URL into **Request URL**. Slack immediately sends a `url_verification` challenge, which n8n answers automatically. You should see **Verified ✅**.
3. Under **Subscribe to bot events** click **Add Bot User Event** and add:
   - `app_mention`
   - `message.channels`
4. Click **Save Changes** (and reinstall if prompted).

### Step 9: Invite the bot to your channels

In Slack, in each channel:

```
/invite @pharma-agent
```

Create and invite the bot to:

- `#ask-pharma-agent` (questions and answers)
- `#pharma-agent-risk1` (risk alerts)

### Step 10: Ask a question

Post in `#ask-pharma-agent`, e.g. `What are the main contraindications of Warfarin?`. The reply is posted by the bot after the agent finishes (it may take several seconds).

---

## 12. Slack Credentials & OAuth Explained

### 12.1 Token types

| Token | Prefix | Purpose | Used in this project? |
|---|---|---|---|
| **Bot User OAuth Token** | `xoxb-` | Actions performed by the bot (read events, post messages). | ✅ Yes |
| **User OAuth Token** | `xoxp-` | Actions on behalf of a person. | ❌ No |
| **App-Level Token** | `xapp-` | Socket Mode connections. | ❌ No (we use HTTP webhooks) |
| **Signing Secret** | (hex string) | Verifies inbound requests. | Recommended |
| **Client ID / Secret** | | OAuth2 code exchange. | Only if you choose OAuth2 in n8n |

### 12.2 What "Install to Workspace" actually does (OAuth 2.0 v2)

1. The app sends the user to Slack's authorize page with its **Client ID**, the requested **scopes** and a **redirect URL**.
2. The workspace admin/user clicks **Allow**.
3. Slack redirects back with a temporary **code**.
4. The app exchanges the code (plus **Client ID** and **Client Secret**) at Slack's `oauth.v2.access` endpoint.
5. Slack returns the **bot token (`xoxb-`)**.

When you click **Install to Workspace** in your own app's dashboard, Slack performs this flow for you and shows the resulting token directly, so you do **not** need a redirect URL for this project.

### 12.3 Optional: n8n "Slack OAuth2 API" credential

Use this only if you prefer n8n to run the OAuth flow itself:

1. In n8n create a credential **Slack OAuth2 API**.
2. Copy the **OAuth Redirect URL** displayed in the dialog (it ends with `/rest/oauth2-credential/callback`).
3. In Slack → **OAuth & Permissions → Redirect URLs → Add New Redirect URL**, paste it and **Save URLs**. Slack expects an HTTPS URL, which is another reason to use the tunnel.
4. Paste **Client ID** and **Client Secret** into n8n.
5. Enter the scopes (space-separated) and click **Connect my account**, then **Allow**.

For a single-workspace project, the **Access Token** method (Step 6) is simpler and equally valid.

### 12.4 Request signing

Slack signs every event request with the headers `X-Slack-Signature` and `X-Slack-Request-Timestamp` (HMAC-SHA256 of the timestamp and raw body with your **Signing Secret**). The receiver recomputes the signature and rejects mismatches, which prevents anyone who discovers your webhook URL from forging events.

---

## 13. Building the n8n Workflows

Two workflows are provided in `n8n_workflows/`. Import them with **Workflows → ⋯ → Import from File**, then re-select credentials in every Slack/OpenAI node and **Publish** each workflow.

### 13.1 Workflow 1: "qa pharma agent in slack workflow"

```
Slack Trigger → If1 ──true──► Edit Fields → HTTP Request → Edit Fields1 → Send a message
                  └─false──► (ignored)
```

| Node | Configuration |
|---|---|
| **Slack Trigger** | Credential: Slack API. *Trigger on*: **Any Event** (or Bot/App Mention). Enable **Watch Whole Workspace** if you want all channels. Its Production URL goes into Slack Event Subscriptions. |
| **If1** | **Loop guard.** Continue only when the event is a real user message: e.g. `bot_id` is empty **and** text is not empty. Without this, the bot answers its own messages forever. |
| **Edit Fields** | Extract the question from the event, e.g. `question = {{ $json.text }}` (remove the `<@BOTID>` mention if present). |
| **HTTP Request** | Method **POST**, URL `http://host.docker.internal:8000/ask`, body type **JSON**: `{ "question": "{{ $json.question }}" }`. Increase **Timeout** (agent calls can take 30–120 s). |
| **Edit Fields1** | Build the text to post, e.g. `reply = {{ $json.answer }}`. |
| **Send a message** | Credential: Slack API. *Send message to*: channel, chosen by ID or by expression `{{ $('Slack Trigger').item.json.channel }}`. *Message text*: `{{ $json.reply }}`. |

Slack formatting tip: Slack uses `*bold*`, whereas the agent writes Markdown `**bold**`. Convert with a small Code node or accept the difference.

### 13.2 Workflow 2: "risk pharma agent message workflow"

```
Webhook (POST) → If ──true──► Message a model (OpenAI) → Send a message
                   └─false──► (ignored: low risk)
```

| Node | Configuration |
|---|---|
| **Webhook** | HTTP Method **POST**, Path e.g. `pharma-risk`. Its Production URL is `N8N_WEBHOOK_URL` in `.env`. |
| **If** | Condition: `{{ $json.body.risk }}` **equals** `high`. (Webhook payloads arrive under `body`.) |
| **Message a model** | OpenAI node, operation *Message a model*, with an OpenAI credential. Prompt example: *"Write a short Slack risk alert for a pharmacist. Question: `{{ $json.body.question }}`. Critic review: `{{ $json.body.review }}`. Keep it under 120 words."* |
| **Send a message** | Credential: Slack API. Channel: `#pharma-agent-risk1`. Text: the model's response text. |

**Payload sent by FastAPI** (`risk.py`):

```json
{
  "question": "…",
  "answer": "…",
  "sources": ["dailymed-warfarin.md"],
  "plan": "1. …",
  "review": "VERDICT: REVISE …",
  "evidence": "…up to 20000 chars…",
  "risk": "high"
}
```

The event is posted for **every** question; the **If** node decides whether an alert is sent. Avoid forwarding `evidence` to Slack (it can be very long).

> The n8n workflow list shows a yellow dot / **Publish** button when a workflow has unpublished changes. Publish it, otherwise the Production URL returns 404.

---

## 14. Recommended Startup Order

1. **Tunnel** (terminal 1):
   ```bash
   cloudflared tunnel --url http://localhost:5678
   ```
   Copy the public URL.
2. **n8n** (terminal 2): `docker run … -e WEBHOOK_URL=<PUBLIC_TUNNEL_URL> …` (see [Section 9](#9-running-n8n-with-docker-command-reference)).
3. **Backend** (terminal 3, venv active):
   ```powershell
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. In n8n: select credentials, **Publish** both workflows, copy the risk webhook Production URL into `.env` (`N8N_WEBHOOK_URL`) and restart the backend if you changed it.
5. In Slack: paste the Slack Trigger Production URL into **Event Subscriptions** (only needed when the tunnel URL changed).
6. Test from the Web UI, then from Slack.

Shutdown: `Ctrl+C` in the tunnel/backend terminals and `docker stop n8n`.

---

## 15. API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /ask`

Runs the full multi-agent pipeline. `question`: 1–2000 characters.

```json
{ "question": "What are the main contraindications of Warfarin?" }
```

Response `200`:

```json
{
  "answer": "Warfarin is contraindicated in: …(dailymed-warfarin.md)\n\nDisclaimer: …",
  "sources": ["dailymed-warfarin.md"],
  "plan": "1. Search contraindications for Warfarin.\n2. Summarize findings.",
  "review": "VERDICT: PASS\nUNSUPPORTED CLAIMS: none\n…",
  "risk": "low"
}
```

Errors: `503` agent not initialized (check startup logs), `500` error while answering.

### `POST /eval`

```json
{
  "question": "What is the interaction between Sertraline and Warfarin?",
  "context": "Concomitant use of sertraline and warfarin may potentiate bleeding risk…",
  "answer": "Combining Sertraline and Warfarin increases risk of bleeding.",
  "reference_answer": "optional gold answer for string similarity"
}
```

| Field in response | Meaning |
|---|---|
| `faithfulness` | LLM output (JSON text) with `faithfulness_score` (0–1) and `ungrounded_claims`. |
| `context_precision` | LLM-produced score 0–1 for how many passages were relevant. |
| `string_similarity` | `difflib.SequenceMatcher` ratio versus `reference_answer` (null if not provided). |

---

## 16. Testing

PowerShell:

```powershell
# Health
Invoke-RestMethod http://localhost:8000/health

# Ask
$body = @{ question = "What are the main contraindications of Warfarin?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -ContentType "application/json" -Body $body

# Evaluate
$eval = @{
  question = "What is the interaction between Sertraline and Warfarin?"
  context  = "Concomitant use of sertraline and warfarin may potentiate bleeding risk..."
  answer   = "Combining Sertraline and Warfarin increases risk of bleeding."
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/eval -Method Post -ContentType "application/json" -Body $eval
```

Test the risk webhook directly (bypassing the agent):

```powershell
$payload = @{ question = "test"; review = "VERDICT: REVISE"; risk = "high"; answer = "test" } | ConvertTo-Json
Invoke-RestMethod -Uri <N8N_RISK_WEBHOOK_URL> -Method Post -ContentType "application/json" -Body $payload
```

Check each layer in this order: **API → n8n workflow 1 → Slack event → risk webhook → Slack alert**. Use the n8n **Executions** tab to see exactly where a run stops.

---

## 17. Command Cheat Sheet

| Goal | Command |
|---|---|
| Create venv | `py -3.11 -m venv .venv` |
| Activate venv (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Install dependencies | `pip install -r requirements.txt` |
| Start API | `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` |
| Start tunnel | `cloudflared tunnel --url http://localhost:5678` |
| Start n8n | `docker run -d --name n8n -p 5678:5678 -e WEBHOOK_URL=<PUBLIC_TUNNEL_URL> -e N8N_PROXY_HOPS=1 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n` |
| n8n logs | `docker logs -f n8n` |
| Stop n8n | `docker stop n8n` |
| Restart stopped n8n | `docker start n8n` |
| Remove n8n container | `docker rm n8n` |
| List containers | `docker ps` (running) / `docker ps -a` (all) |
| Optional Compose stack | `docker compose up -d --build` / `docker compose down` |

---

## 18. Troubleshooting

| Symptom | Likely cause & fix |
|---|---|
| Slack says *Your URL didn't respond with the value of the challenge parameter* | Workflow not published, tunnel not running, or wrong URL (Test vs Production). |
| Slack events stopped arriving | The quick-tunnel URL changed. Recreate n8n with the new `WEBHOOK_URL` and update Slack. |
| Bot replies to itself in a loop | Missing/incorrect **If1** guard. Ignore events that have a `bot_id`. |
| Bot does not react to messages | Not invited to the channel, missing event subscription, or scopes added without reinstalling. |
| `channel_not_found` / `not_in_channel` when posting | Invite the bot (`/invite @bot`) or add `chat:write.public`. |
| `invalid_auth` / `token_revoked` | Wrong or regenerated token. Paste the current `xoxb-` token into the n8n credential. |
| n8n HTTP Request: *ECONNREFUSED* on `localhost:8000` | `localhost` is the container. Use `host.docker.internal`. Also start Uvicorn with `--host 0.0.0.0`. |
| n8n HTTP Request times out | Agent runs take a while; raise the node timeout. |
| `503 Agent not ready` | Startup failed (bad API key, wrong base URL/model, embeddings unreachable, empty `data/`). Read the Uvicorn log. |
| `No .md files found` | Check `DATA_DIR` and that files end with `.md`. |
| Risk alerts never arrive | `N8N_WEBHOOK_URL` empty or pointing at the Test URL; risk workflow unpublished; or `risk` is `low`. |
| Pydantic / type errors at import | You are using Python 3.14. Switch to 3.11 or 3.12. |
| PowerShell blocks `Activate.ps1` | Run the `Set-ExecutionPolicy` command from [Section 6](#6-prerequisites). |
| Changed `WEBHOOK_URL` has no effect | You must `docker rm` and recreate the container; env vars are not editable after creation. |

---

## 19. Security Checklist

- 🔒 Never commit `.env`; add it to `.gitignore`.
- 🔑 Treat `xoxb-` tokens, the Signing Secret, Client Secret and your LLM key as passwords. If one leaks, **regenerate it** (Slack: *OAuth & Permissions → Reinstall*, or *Basic Information → Regenerate*).
- 🌐 Anyone who knows a public webhook URL can call it. Keep the Signing Secret configured and consider adding an authentication header to the n8n Webhook node.
- 🧪 Quick tunnels are for development. Use a named tunnel or a proper domain for anything long-lived.
- 🧱 Grant the Slack app only the scopes it needs.
- 🩺 Keep the disclaimer: outputs are not clinical advice.

---

## 20. Known Limitations

- **In-memory indexes.** Chroma is not persisted, so embeddings and the knowledge graph are rebuilt (and LLM/embedding calls are billed) at every startup.
- **No graph disk cache.** The graph lives only in memory; caching it to JSON would speed up restarts.
- **Graph extraction granularity.** Triples are extracted per whole document and may be incomplete for long monographs.
- **Cypher helper not wired.** `generate_few_shot_cypher()` exists in `graph_rag.py` but is not exposed as an agent tool, and the graph is queried by entity name, not by executing Cypher.
- **`/eval` returns raw LLM text** for faithfulness and precision rather than parsed numbers.
- **Latency.** Planner, agent, Critic and Finalizer are sequential LLM calls; answers take several seconds up to a couple of minutes.
- **Retrieval depth.** `TOP_K = 3` with 200-token chunks is deliberately narrow; increase it for broader recall.

---

## ⚕️ Disclaimer

This project is a technical exercise based only on the provided documents. It is **not medical advice**. Consult a qualified healthcare professional.