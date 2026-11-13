import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from app.agent import PharmaAgent
from app.config import MCP_SERVER_URL, STATIC_DIR
from app.eval import RAGEvaluator
from app.risk import assess_risk, notify_n8n

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharma-agent")

MCP_CONNECT_RETRIES = 60
MCP_RETRY_DELAY_S = 3


def _causes(exc: BaseException):
    """Déballe récursivement les ExceptionGroup."""
    subs = getattr(exc, "exceptions", None)
    if subs:
        for s in subs:
            yield from _causes(s)
    else:
        yield exc


async def connect_agent(app: FastAPI) -> None:
    """Wait for the MCP server, load its tools, then build the agent.

    Runs in the background so /health answers immediately while the MCP
    server is still loading the corpus.
    """
    client = MultiServerMCPClient(
        {"pharma": {"transport": "streamable_http", "url": MCP_SERVER_URL}}
    )
    for attempt in range(1, MCP_CONNECT_RETRIES + 1):
        try:
            tools = await client.get_tools()
            app.state.agent = PharmaAgent(tools)
            logger.info("Agent ready with %d MCP tools: %s",
                        len(tools), [t.name for t in tools])
            return
        except Exception as e:
            causes = " | ".join(repr(c) for c in _causes(e))
            logger.info("Waiting for MCP server (%d/%d): %s",
                        attempt, MCP_CONNECT_RETRIES, causes)
            await asyncio.sleep(MCP_RETRY_DELAY_S)
    app.state.agent_error = f"MCP server unreachable at {MCP_SERVER_URL}"
    logger.error(app.state.agent_error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.evaluator = RAGEvaluator()
    task = asyncio.create_task(connect_agent(app))
    yield
    task.cancel()


app = FastAPI(title="Pharmacology Research Agent (MCP + Hybrid RAG + GraphRAG)",
              lifespan=lifespan)


class AskRequest(BaseModel):
    # min_length=1 accepts single-character questions while rejecting empty strings
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
    return {"status": "ok", "agent_ready": hasattr(app.state, "agent")}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, background_tasks: BackgroundTasks) -> AskResponse:
    if not hasattr(app.state, "agent"):
        detail = getattr(app.state, "agent_error", "Agent is starting, retry in a few seconds")
        raise HTTPException(status_code=503, detail=f"Agent not ready: {detail}")

    try:
        result = await app.state.agent.ask(request.question)
    except Exception as e:
        logger.exception("Error while answering")
        raise HTTPException(status_code=500, detail=f"Error while answering: {e}") from e

    result["risk"] = assess_risk(result["review"])

    background_tasks.add_task(notify_n8n, {"question": request.question, **result})

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        plan=result["plan"],
        review=result["review"],
        risk=result["risk"],
    )


@app.post("/eval")
def evaluate_rag(request: EvalRequest) -> dict:
    """RAG Evaluation Endpoint (Context Precision, Faithfulness, String Similarity).

    Declared with `def` on purpose: FastAPI runs it in a thread pool, so the
    blocking LLM calls do not freeze the event loop.
    """
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