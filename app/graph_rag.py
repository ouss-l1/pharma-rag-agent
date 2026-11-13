import hashlib
import json
import logging
import re
from typing import Dict, Any, List

import networkx as nx
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.config import (
    CACHE_DIR,
    CHAT_MODEL,
    GRAPH_CONCURRENCY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger("pharma-mcp")

GRAPH_CACHE_FILE = CACHE_DIR / "graph_cache.json"
GRAPH_KEYWORDS = ["contraindication", "interaction", "warning", "adverse"]


class KnowledgeTriple(BaseModel):
    subject: str = Field(description="Source drug or molecule, e.g., Warfarin, Metformin")
    relation: str = Field(description="Relationship type: CONTRAINDICATED_WITH, INTERACTS_WITH, CAUSES_SIDE_EFFECT, DOSAGE_RULE")
    object: str = Field(description="Target condition, drug, or clinical threshold, e.g., Pregnancy, Aspirin, eGFR < 30")
    effect: str = Field(default="", description="Specific clinical consequence or warning detail")


class DocumentTriples(BaseModel):
    triples: List[KnowledgeTriple]


TRIPLE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a medical Knowledge Graph extractor. Analyze the Markdown text from a drug notice.
        Extract all factual clinical relationships into structured triples:
        - subject: Drug/Molecule name
        - relation: CONTRAINDICATED_WITH | INTERACTS_WITH | CAUSES_SIDE_EFFECT | DOSAGE_RULE
        - object: Target drug, medical condition, or clinical parameter
        - effect: Short clinical detail (or empty if none)

        Extract ONLY relationships explicitly stated in the text."""
    ),
    ("human", "Markdown Text:\n{text}")
])

# NOTE: literal curly braces must be doubled ({{ }}) in a ChatPromptTemplate,
# otherwise they are parsed as template variables.
CYPHER_FEW_SHOT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Cypher query generator for a Pharmacology Knowledge Graph extracted from Markdown notices.
        Schema:
        - Nodes: Drug (name), Condition (name), SideEffect (name)
        - Relationships: CONTRAINDICATED_WITH, INTERACTS_WITH, CAUSES_SIDE_EFFECT

        Few-Shot Examples:
        Q: What is contraindicated with Warfarin?
        Cypher: MATCH (d:Drug {{name: 'Warfarin'}})-[r:CONTRAINDICATED_WITH]->(c:Condition) RETURN d.name, type(r), c.name

        Output ONLY the valid Cypher query code."""
    ),
    ("human", "Q: {question}\nCypher:")
])


def _fingerprint(documents: List[Document], model: str) -> str:
    """Hash of the corpus + model: the cache is reused only if nothing changed."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    for doc in sorted(documents, key=lambda d: d.metadata.get("source", "")):
        h.update(doc.metadata.get("source", "").encode("utf-8"))
        h.update(doc.page_content.encode("utf-8"))
    return h.hexdigest()


class GraphRAGEngine:
    """Dynamic Knowledge Graph engine that extracts triples automatically from Markdown documents."""

    def __init__(self, documents: List[Document] = None):
        self.graph = nx.MultiDiGraph()
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            max_retries=6,
        )
        self.cypher_generator = CYPHER_FEW_SHOT_PROMPT | self.llm | StrOutputParser()

        if documents:
            self.build_graph_from_documents(documents)

    # ------------------------------------------------------------------ cache
    def _load_cache(self, fingerprint: str) -> bool:
        try:
            if not GRAPH_CACHE_FILE.exists():
                return False
            data = json.loads(GRAPH_CACHE_FILE.read_text(encoding="utf-8"))
            if data.get("fingerprint") != fingerprint:
                logger.info("Graph cache is outdated, rebuilding.")
                return False
            for e in data["edges"]:
                self.graph.add_edge(
                    e["src"], e["dst"],
                    relation=e["relation"], effect=e["effect"], source=e["source"],
                )
            logger.info("Graph loaded from cache: %d edges", self.graph.number_of_edges())
            return True
        except Exception as e:
            logger.warning("Could not read graph cache: %r", e)
            self.graph.clear()
            return False

    def _save_cache(self, fingerprint: str) -> None:
        try:
            GRAPH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            edges = [
                {"src": u, "dst": v, **{k: d.get(k, "") for k in ("relation", "effect", "source")}}
                for u, v, d in self.graph.edges(data=True)
            ]
            GRAPH_CACHE_FILE.write_text(
                json.dumps({"fingerprint": fingerprint, "edges": edges}, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Graph cache saved (%d edges)", len(edges))
        except Exception as e:
            logger.warning("Could not write graph cache: %r", e)

    # ------------------------------------------------------------------ build
    def build_graph_from_documents(self, documents: List[Document]):
        """Parses Markdown documents and populates NetworkX edges using LLM triple extraction."""
        fingerprint = _fingerprint(documents, CHAT_MODEL)
        if self._load_cache(fingerprint):
            return

        candidates = [
            d for d in documents
            if any(kw in d.page_content.lower() for kw in GRAPH_KEYWORDS)
        ]
        logger.info("Graph: extracting triples from %d/%d documents (concurrency=%d)",
                    len(candidates), len(documents), GRAPH_CONCURRENCY)

        structured_llm = self.llm.with_structured_output(DocumentTriples)
        extractor_chain = TRIPLE_EXTRACTION_PROMPT | structured_llm

        results = extractor_chain.batch(
            [{"text": d.page_content} for d in candidates],
            config={"max_concurrency": GRAPH_CONCURRENCY},
            return_exceptions=True,
        )

        failures = 0
        for doc, res in zip(candidates, results):
            source_file = doc.metadata.get("source", "unknown.md")
            if isinstance(res, Exception) or res is None:
                failures += 1
                logger.warning("Triple extraction failed for %s: %r", source_file, res)
                continue
            for t in res.triples:
                src = t.subject.strip()
                dst = t.object.strip()
                if not src or not dst:
                    continue
                self.graph.add_edge(
                    src,
                    dst,
                    relation=t.relation.upper(),
                    effect=t.effect,
                    source=source_file,
                )

        # Only cache a complete graph, so a failed run is retried next time.
        if failures == 0:
            self._save_cache(fingerprint)
        else:
            logger.warning("%d document(s) failed: graph cache NOT saved.", failures)

    # ------------------------------------------------------------------ query
    def query_graph_relations(self, entity: str) -> str:
        """Retrieves graph edges where the entity is subject or object (partial, case-insensitive match)."""
        needle = entity.strip().lower()
        if not needle:
            return "Empty entity."

        results = set()
        for u, v, data in self.graph.edges(data=True):
            if needle in u.lower() or needle in v.lower():
                eff = data.get("effect", "")
                eff_str = f" ({eff})" if eff else ""
                rel = data.get("relation", "RELATED_TO")
                src = data.get("source", "unknown.md")
                results.add(f"({u}) -[{rel}]-> ({v}){eff_str} [{src}]")

        if not results:
            return f"No direct graph relationships found for '{entity}' in the corpus."

        return "Graph RAG Results:\n" + "\n".join(sorted(results))

    def validate_cypher(self, cypher_query: str) -> bool:
        """Validates generated Cypher queries against destructive operations."""
        forbidden_keywords = ["DELETE", "DROP", "CREATE", "SET", "REMOVE"]
        if any(keyword in cypher_query.upper() for keyword in forbidden_keywords):
            return False
        return cypher_query.strip().upper().startswith("MATCH")

    def generate_few_shot_cypher(self, question: str) -> Dict[str, Any]:
        """Generates a Cypher query using few-shot prompts and checks validation."""
        raw_cypher = self.cypher_generator.invoke({"question": question})
        cleaned_cypher = re.sub(r"```cypher|```", "", raw_cypher).strip()
        is_valid = self.validate_cypher(cleaned_cypher)
        return {
            "query": cleaned_cypher,
            "valid": is_valid,
            "graph_output": self.query_graph_relations(question) if is_valid else "Invalid Cypher query syntax"
        }