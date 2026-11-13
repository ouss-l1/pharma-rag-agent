# Pharma Research Agent

A small agentic RAG assistant (LangChain + LangGraph + FastAPI) over a local corpus of drug documents.

## Architecture
1. **Plan** – The LLM splits the question into sub-questions (structured output).
2. **Retrieve + Tools** – A ReAct agent chooses among dynamically injected tools (`search_drug_documents`, `get_drug_section`, `lookup_glossary`, `list_available_drugs`). Search uses an in-memory Chroma vector store.
3. **Review** – A critic checks the notes, flags unsupported / clinically ambiguous claims, and rewrites the answer. High-risk answers trigger a background notification to an n8n webhook.
4. **Answer** – Cited with source file names + not-medical-advice disclaimer.

## Run locally
    cp .env.example .env   # fill in your credentials
    pip install -r requirements.txt
    uvicorn app.main:app --reload
Open http://localhost:8000

## Run with Docker
    docker build -t agent . && docker run -p 8000:8000 --env-file .env agent

## API
- `GET /health` → `{"status": "ok"}`
- `POST /ask` with `{"question": "..."}` → `{answer, sources, plan, review, risk}`

## Limitations
- Answers are only as good as the provided corpus files; the agent explicitly states when the documents don't cover a question.