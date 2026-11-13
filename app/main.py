import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import PharmaAgent
from app.config import DATA_DIR, STATIC_DIR
from app.eval import RAGEvaluator
from app.graph_rag import GraphRAGEngine
from app.rag import KnowledgeBase
from app.risk import assess_risk, notify_n8n

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharma-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Markdown Knowledge Base, Dynamic Graph Engine, Evaluator, and Agent."""
    try:
        kb = KnowledgeBase(DATA_DIR)
        graph_engine = GraphRAGEngine(documents=kb.raw_documents)
        app.state.evaluator = RAGEvaluator()
        app.state.agent = PharmaAgent(kb, graph_engine)
        logger.info(
            "Markdown Hybrid Agent Ready: %d chunks & %d graph edges indexed", 
            kb.chunk_count, 
            graph_engine.graph.number_of_edges()
        )
    except Exception as e:
        logger.exception("Could not initialize the agent")
        app.state.agent_error = str(e)
    yield


app = FastAPI(title="Pharmacology Research Agent (Markdown & Graph RAG)", lifespan=lifespan)


class AskRequest(BaseModel):
    # Set min_length=1 to accept single-character questions (e.g. "?", "a") while rejecting empty strings
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    plan: str
    review: str
    risk: str


class EvalRequest(BaseModel):
    question: str
    answer: str
    context: str
    reference_answer: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, background_tasks: BackgroundTasks) -> AskResponse:
    if not hasattr(app.state, "agent"):
        error_msg = getattr(app.state, "agent_error", "Unknown startup error")
        raise HTTPException(status_code=503, detail=f"Agent not ready: {error_msg}")
    
    try:
        result = app.state.agent.ask(request.question)
    except Exception as e:
        logger.exception("Error while answering")
        raise HTTPException(status_code=500, detail=f"Error while answering: {e}") from e

    result["risk"] = assess_risk(result["review"])
    
    background_tasks.add_task(
        notify_n8n, {"question": request.question, **result}
    )
    
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        plan=result["plan"],
        review=result["review"],
        risk=result["risk"],
    )


@app.post("/eval")
def evaluate_rag(request: EvalRequest) -> dict:
    """RAG Evaluation Endpoint (Context Precision, Faithfulness, String Similarity)."""
    if not hasattr(app.state, "evaluator"):
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    evaluator: RAGEvaluator = app.state.evaluator
    return evaluator.evaluate_rag_triplet(
        question=request.question,
        context=request.context,
        answer=request.answer,
        reference=request.reference_answer,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")