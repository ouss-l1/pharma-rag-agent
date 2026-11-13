import re
from typing import Dict, Any, List
import networkx as nx
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL


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

CYPHER_FEW_SHOT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Cypher query generator for a Pharmacology Knowledge Graph extracted from Markdown notices.
        Schema:
        - Nodes: Drug (name), Condition (name), SideEffect (name)
        - Relationships: CONTRAINDICATED_WITH, INTERACTS_WITH, CAUSES_SIDE_EFFECT

        Few-Shot Examples:
        Q: What is contraindicated with Warfarin?
        Cypher: MATCH (d:Drug {name: 'Warfarin'})-[r:CONTRAINDICATED_WITH]->(c:Condition) RETURN d.name, type(r), c.name

        Output ONLY the valid Cypher query code."""
    ),
    ("human", "Q: {question}\nCypher:")
])


class GraphRAGEngine:
    """Dynamic Knowledge Graph engine that extracts triples automatically from Markdown documents."""

    def __init__(self, documents: List[Document] = None):
        self.graph = nx.DiGraph()
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.cypher_generator = CYPHER_FEW_SHOT_PROMPT | self.llm | StrOutputParser()
        
        if documents:
            self.build_graph_from_documents(documents)

    def build_graph_from_documents(self, documents: List[Document]):
        """Dynamically parses Markdown documents and populates NetworkX edges using LLM triple extraction."""
        structured_llm = self.llm.with_structured_output(DocumentTriples)
        extractor_chain = TRIPLE_EXTRACTION_PROMPT | structured_llm

        for doc in documents:
            source_file = doc.metadata.get("source", "unknown.md")
            if any(kw in doc.page_content.lower() for kw in ["contraindication", "interaction", "warning", "adverse"]):
                try:
                    res: DocumentTriples = extractor_chain.invoke({"text": doc.page_content})
                    for t in res.triples:
                        src = t.subject.strip().title()
                        dst = t.object.strip().title()
                        self.graph.add_edge(
                            src,
                            dst,
                            relation=t.relation.upper(),
                            effect=t.effect,
                            source=source_file
                        )
                except Exception:
                    continue

    def query_graph_relations(self, entity: str) -> str:
        """Retrieves exact graph edges and relational links for an entity."""
        entity_clean = entity.strip().title()
        results = []

        if self.graph.has_node(entity_clean):
            for neighbor in self.graph.neighbors(entity_clean):
                edge_data = self.graph.get_edge_data(entity_clean, neighbor)
                rel = edge_data.get("relation", "RELATED_TO")
                src = edge_data.get("source", "dailymed-notice.md")
                eff = edge_data.get("effect", "")
                eff_str = f" ({eff})" if eff else ""
                results.append(f"({entity_clean}) -[{rel}]-> ({neighbor}){eff_str} [{src}]")

        for u, v, data in self.graph.edges(data=True):
            if v.lower() == entity.lower():
                rel = data.get("relation", "RELATED_TO")
                src = data.get("source", "dailymed-notice.md")
                eff = data.get("effect", "")
                eff_str = f" ({eff})" if eff else ""
                results.append(f"({u}) -[{rel}]-> ({v}){eff_str} [{src}]")

        if not results:
            return f"No direct graph relationships found for '{entity}' in the corpus."

        return "Graph RAG Results:\n" + "\n".join(results)

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