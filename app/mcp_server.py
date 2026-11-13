"""MCP server exposing the pharmacology retrieval tools.

Run with:  python -m app.mcp_server
Endpoint:  http://127.0.0.1:8001/mcp  (streamable HTTP)

The knowledge base (embeddings, BM25) and the knowledge graph are built ONCE
when this process starts, then shared by every MCP client.
"""
import logging

from mcp.server.fastmcp import FastMCP

from app.config import DATA_DIR, MCP_HOST, MCP_PORT
from app.graph_rag import GraphRAGEngine
from app.rag import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharma-mcp")


class _IgnoreClosedSSE(logging.Filter):
    """Hide the harmless error logged when an MCP session closes normally."""

    def filter(self, record: logging.LogRecord) -> bool:
        if "standalone SSE writer" in record.getMessage():
            return False
        if record.exc_info and record.exc_info[0] is not None:
            return record.exc_info[0].__name__ != "ClosedResourceError"
        return True


def _silence_closed_sse() -> None:
    """Apply the filter on the logger AND on every handler (belt and braces)."""
    flt = _IgnoreClosedSSE()
    logging.getLogger("mcp.server.streamable_http").addFilter(flt)
    for handler in logging.getLogger().handlers:
        handler.addFilter(flt)


# Heavy initialisation, done once at startup
kb = KnowledgeBase(DATA_DIR)
graph_engine = GraphRAGEngine(documents=kb.raw_documents)
logger.info(
    "MCP server ready: %d chunks & %d graph edges indexed",
    kb.chunk_count,
    graph_engine.graph.number_of_edges(),
)

# stateless_http=True: no session kept between calls, which matches how
# langchain-mcp-adapters works (one short session per call).
mcp = FastMCP("pharma-tools", host=MCP_HOST, port=MCP_PORT, stateless_http=True)

# Apply AFTER FastMCP is created, because FastMCP configures logging itself.
_silence_closed_sse()


@mcp.tool()
def search_hybrid(query: str) -> str:
    """Hybrid Sparse BM25 + Dense Vector search over Markdown drug documents.
    Use it for contraindications, dosages, and general queries."""
    return kb.search_hybrid(query)


@mcp.tool()
def search_bm25_exact(query: str) -> str:
    """Sparse BM25 search for exact medical keyword and dosage matching in Markdown files."""
    return kb.search_bm25(query)


@mcp.tool()
def query_knowledge_graph(entity: str) -> str:
    """Query the relational Knowledge Graph for structured drug contraindications,
    interactions and dosage rules of a drug or condition."""
    return graph_engine.query_graph_relations(entity)


@mcp.tool()
def get_drug_section(drug: str, section: str) -> str:
    """Return a full section of one drug's Markdown document (e.g. drug='warfarin',
    section='contraindications')."""
    return kb.get_section(drug, section)


@mcp.tool()
def lookup_glossary(term: str) -> str:
    """Look up a medical term in the Markdown glossary."""
    return kb.glossary(term)


@mcp.tool()
def list_available_drugs() -> str:
    """List the available Markdown drug files in the corpus."""
    return kb.list_drugs()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")