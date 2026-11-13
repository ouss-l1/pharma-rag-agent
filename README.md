# 💊 Advanced Pharmacology GenAI Agent

**Enterprise-Grade Multi-Agent RAG with MCP, GraphRAG, and Human-in-the-Loop Safety Audits**

## 🌟 Strategic Value & Project Overview

Standard RAG (Retrieval-Augmented Generation) applications often fall short in healthcare domains due to hallucination risks and a lack of accountability. This project transcends basic RAG by implementing a **Zero-Trust Multi-Agent Architecture**.

Designed for clinical pharmacology (ingesting FDA labels and drug monographs), this system does not just retrieve and generate; it **critically audits its own reasoning**, structures complex clinical relationships into a dynamic graph, and enforces a **Human-in-the-Loop (HITL)** governance model via an event-driven n8n and Slack pipeline.

### 🏆 Key Innovations & Architectural Strengths

1. **Model Context Protocol (MCP) Decoupling:** Tools (Graph queries, BM25, Chroma) are isolated in a dedicated FastMCP server. This allows the reasoning agent to access the Knowledge Base over streamable HTTP, mirroring microservice architectures used in top-tier tech organizations and future-proofing the system for distributed scaling.

2. **Dynamic Knowledge Graph (GraphRAG):** Medical queries often require multi-hop reasoning (e.g., comparing shared contraindications). The system dynamically extracts clinical triples (`CONTRAINDICATED_WITH`, `INTERACTS_WITH`) from raw Markdown into an in-memory NetworkX graph, vastly outperforming pure vector-similarity searches on relational data.

3. **Multi-Agent Safety Pipeline (ReAct + Critic):** A primary ReAct agent drafts the answer, but it cannot output directly to the user. A strict **Critic Node** intercepts the draft, aggressively verifying every sentence against the retrieved Markdown chunks to flag unsupported or clinically ambiguous claims.

4. **Asynchronous Risk Routing:** Using n8n and Cloudflare, high-risk queries are silently routed to a dedicated compliance channel without blocking the user's immediate HTTP response, ensuring both speed for the user and safety for the organization.

## 🧠 The Engine: `gpt-5.6-terra`

Powered by the `gpt-5.6-terra` model (accessed via a LiteLLM proxy), the system leverages its advanced capabilities perfectly:

* **Strategic Decomposition (Planner):** Cleanly breaks down compound, multi-drug questions into actionable search steps.
* **Schema Adherence:** Flawlessly outputs structured JSON required for GraphRAG extraction and evaluation metrics.
* **Clinical Nuance (Critic):** Demonstrates high-level contextual understanding, capable of distinguishing between absolute contraindications and conditional warnings during the safety audit loop.

## 🔌 FastMCP & Tool Decoupling (Model Context Protocol)

Instead of tightly coupling heavy databases (ChromaDB, NetworkX) into the main FastAPI web server, this project utilizes **Anthropic's Model Context Protocol (MCP)** via the `FastMCP` framework. 

The MCP Server (`mcp_server.py`) acts as an independent microservice that loads the heavy Knowledge Base into memory exactly once. The LangChain ReAct agent then connects to it dynamically to utilize the following exposed tools:

* 🔍 **`search_hybrid`**: Combines Dense Vector search (Chroma) for semantic meaning and Sparse Keyword search (BM25) using Reciprocal Rank Fusion (RRF).
* 🎯 **`search_bm25_exact`**: Dedicated sparse search to ensure exact numerical doses, clinical thresholds, and specific drug names are never missed.
* 🕸️ **`query_knowledge_graph`**: Executes Cypher-style logical queries against the extracted NetworkX graph for relational data (e.g., "What conditions are contraindicated with X?").
* 📑 **`get_drug_section`**: Extracts whole specific sections (e.g., "Warnings") from the raw Markdown to provide deep context without vector fragmentation.

**Why this matters:** This decoupling allows the FastAPI web server to remain lightning-fast and stateless. The MCP server does the heavy lifting, ensuring enterprise-level memory efficiency and scalability.

## 🏗️ Architecture Deep-Dive: Dual-Entry & Slack Governance

This system breaks out of the standard web-only UI by implementing a **Dual-Entry Architecture**, treating Slack as a first-class citizen for both querying and governance.

```
[User] ──(Web UI)──> [FastAPI Backend] ──(Critic Audit)──> [n8n Webhook] ──> [Slack: #pharma-agent-risk1]
                         ▲
[User] ──(Slack)───> [n8n Workflow] ──(Cloudflare Tunnel)──> [FastAPI Backend]
```

### The Two Pillars of the Slack Integration

To operationalize this, the workspace relies on two distinct Slack channels, each with a specific enterprise purpose:

* 🩺 **`#ask-pharma-agent` (The Front-Line Interface):**
  Designed for end-users (e.g., researchers, clinicians). Users can tag the bot directly in this channel to ask complex pharmacological questions. The n8n workflow captures the event, queries the FastAPI backend, and returns a fully cited, formatted Markdown answer back to the Slack thread in real-time.

* 🛡️ **`#pharma-agent-risk1` (The Governance & Audit Channel):**
  Designed for back-office compliance and medical reviewers. If the internal Critic Node flags an answer for missing caveats or unverified claims, the system triggers a background task. n8n catches this webhook and instantly pushes a highly structured "Risk Alert" to this channel, detailing *Why* it was flagged and *What to Check* in the source files, enabling instant Human-in-the-Loop oversight.

### 🌐 The Crucial Role of Cloudflare Tunnels

To allow the local Dockerized n8n instance to securely receive live events from Slack's public servers, the system uses **Cloudflare Tunnels (`cloudflared`)**. This creates a secure, outbound-only connection, exposing *only* the n8n webhook port to the internet without requiring router port-forwarding or exposing the core FastAPI backend.

## 🚀 Step-by-Step Setup Guide

### 1. Local Environment & FastAPI Setup

1. **Clone and create a virtual environment:**
   ```bash
   git clone <your-repo-url>
   cd pharma-rag-agent
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Data & Env Variables:**
   * Place your `.md` drug files in a `data/` folder at the root.
   * Create a `.env` file based on the provided credentials. Leave `N8N_WEBHOOK_URL` blank for now:
   ```env
   OPENAI_API_KEY=your_key
   OPENAI_BASE_URL=https://proxy.syntax.com/v1
   OPENAI_MODEL=gpt-5.6-terra
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   DATA_DIR=./data
   SPLIT_METHOD=markdown
   N8N_WEBHOOK_URL=
   ```

### 2. Expose n8n with Cloudflare

1. **Install Cloudflared** (Example for Windows):
   ```bash
   winget install Cloudflare.cloudflared
   ```
2. **Start the tunnel** targeting port 5678 (n8n's default port):
   ```bash
   cloudflared tunnel --url http://localhost:5678
   ```
3. **Copy the URL:** Look for the URL ending in `.trycloudflare.com` in your terminal output. Keep this terminal window running.

### 3. n8n Docker Deployment

Run n8n, passing the Cloudflare URL so n8n knows its public webhook address:

```bash
docker run -d --name n8n -p 5678:5678 \
  -e WEBHOOK_URL=https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com \
  -e N8N_PROXY_HOPS=1 \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n
```

### 4. Slack App Configuration

1. Go to [Slack API Dashboard](https://api.slack.com/apps) and click **Create New App** (From Scratch).
2. Go to **OAuth & Permissions** and add the following **Bot Token Scopes**:
   * `app_mentions:read` (To hear when someone tags the bot)
   * `chat:write` (To reply with answers)
   * `chat:write.public` (To post risk alerts in channels)
3. Click **Install to Workspace** and copy the **Bot User OAuth Token** (`xoxb-...`).
4. **Create Channels:** In your Slack workspace, create `#ask-pharma-agent` and `#pharma-agent-risk1`.
5. **Invite the Bot:** Go into both channels and type `/invite @YourBotName`.

### 5. Configuring the 2-Way Webhooks (n8n ↔ Slack ↔ Backend)

1. Open `http://localhost:5678` in your browser.
2. Go to **Credentials**, add a new **Slack API** credential, and paste your `xoxb-...` token.
3. Import the two provided JSON workflows: `slack_pharma_qa_workflow.json` and `pharma_risk_alerts_workflow.json`.

**A. Link Slack to n8n (QA Workflow)**
1. Open the QA workflow (`slack_pharma_qa_workflow`).
2. Double-click the **Slack Trigger** node and copy the **Production URL**.
3. Go back to your Slack App Dashboard -> **Event Subscriptions**.
4. Toggle **Enable Events** to On.
5. Paste the **Production URL** you just copied into the **Request URL** field (Slack will verify it automatically).
6. Under *Subscribe to bot events*, add `app_mention` and `message.channels`. Click **Save Changes**.
7. In the n8n QA workflow, open the **HTTP Request** node and ensure the URL is `http://host.docker.internal:8000/ask`.
8. Open the **Send a Message** node and select the `#ask-pharma-agent` channel.

**B. Link the Backend to n8n (Risk Workflow)**
1. Open the Risk Alerts workflow (`pharma_risk_alerts_workflow`).
2. Double-click the **Webhook** node and copy the **Production URL** (it should look like `https://...trycloudflare.com/webhook/pharma-risk`).
3. Open your FastAPI `.env` file and paste this URL:
   ```env
   N8N_WEBHOOK_URL=https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/webhook/pharma-risk
   ```
4. Open the **Send a Message** node and select the `#pharma-agent-risk1` channel.

**Activate both workflows in the top right corner of the n8n UI!**

### 6. Run the Agent (Two Processes)

Because the system uses an advanced decoupled architecture, you need to run the MCP tools server and the FastAPI web server side-by-side.

**Terminal 1: Start the MCP Server**
```bash
python -m app.mcp_server
```
*(On startup, this server builds the NetworkX Graph and ChromaDB embeddings. It caches them in `graph_cache.json` for instant subsequent startups. Leave it running.)*

**Terminal 2: Start the FastAPI Server**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*(The FastAPI server will automatically connect to the MCP server to provision the LangChain ReAct agent).*

## 🧪 Built-in Evaluation Engine

The system includes a dedicated `/eval` endpoint using LLM-as-a-judge patterns to score its own performance, measuring:

* **Context Precision:** Ratio of relevant vs. irrelevant retrieved chunks.
* **Faithfulness:** Ensures 100% of the generated answer is strictly grounded in the retrieved context.
* **String Similarity (Levenshtein):** Compares output against hidden reference answers.

## 📄 License & Disclaimer

This is a technical GenAI demonstration for the Syntax Bootcamp assignment. The outputs generated by this system do not constitute medical advice.