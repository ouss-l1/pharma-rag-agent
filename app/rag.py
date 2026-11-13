import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    Language,
)

# Import standard d'EnsembleRetriever

from langchain_classic.retrievers.ensemble import EnsembleRetriever

from app.config import (
    BM25_WEIGHT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SPLIT_METHOD,
    TOP_K,
)

HEADING = re.compile(r"^(#{1,6})\s+(.*)")


@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        check_embedding_ctx_length=False,
    )


def drug_name(path: Path) -> str:
    """Extract drug identifier from Markdown filename (e.g., 'dailymed-warfarin.md' -> 'warfarin')."""
    stem = path.stem.lower()
    return stem.split("-", 1)[1] if "-" in stem else stem


class KnowledgeBase:
    """Hybrid RAG pipeline strictly built for Markdown (.md) document corpora."""

    def __init__(self, data_dir: Path):
        paths = sorted(Path(data_dir).glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No .md files found in directory: {data_dir}")

        self.texts: dict[str, str] = {}
        self.drug_files: dict[str, list[str]] = {}
        documents: List[Document] = []

        # 1. Load Markdown documents
        for path in paths:
            docs = TextLoader(str(path), encoding="utf-8").load()
            drug = drug_name(path)
            for doc in docs:
                doc.metadata["source"] = path.name
                doc.metadata["drug"] = drug
            documents.extend(docs)
            self.texts[path.name] = path.read_text(encoding="utf-8")
            if drug != "glossary":
                self.drug_files.setdefault(drug, []).append(path.name)

        # Retain raw unchunked documents for dynamic Graph RAG extraction
        self.raw_documents = documents

        # 2. Split Markdown using Markdown-aware or Token splitters
        chunks = self._split_documents(documents)

        # Prefix every chunk with its drug so it is never "orphaned"
        for chunk in chunks:
            drug = chunk.metadata.get("drug", "unknown")
            chunk.page_content = f"[Drug: {drug}]\n{chunk.page_content}"

        # 3. Dense Index (Chroma Vector Store)
        self.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=get_embeddings()
        )
        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K}
        )

        # 4. Sparse Index (BM25 Retriever)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = TOP_K

        # 5. Hybrid Ensemble Retriever (RRF - Reciprocal Rank Fusion)
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[BM25_WEIGHT, DENSE_WEIGHT],
        )

        self.chunk_count = len(chunks)

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        if SPLIT_METHOD == "token":
            splitter = TokenTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
        elif SPLIT_METHOD == "markdown":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        return splitter.split_documents(documents)

    def search_hybrid(self, query: str) -> str:
        """Executes Hybrid (BM25 Sparse + Chroma Dense) search."""
        docs = self.hybrid_retriever.invoke(query)
        if not docs:
            return "No relevant passages found via Hybrid Search."
        return "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        )

    def search_bm25(self, query: str) -> str:
        """Executes BM25 sparse keyword search."""
        docs = self.bm25_retriever.invoke(query)
        if not docs:
            return "No exact matches found via BM25."
        return "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        )

    def search(self, query: str) -> str:
        return self.search_hybrid(query)

    def list_drugs(self) -> str:
        lines = [f"- {drug}: {', '.join(files)}" for drug, files in sorted(self.drug_files.items())]
        return "Available drugs and files:\n" + "\n".join(lines)

    def get_section(self, drug: str, section: str) -> str:
        files = self.drug_files.get(drug.strip().lower())
        if not files:
            return f"Unknown drug '{drug}'. " + self.list_drugs()

        results, headings = [], []
        for name in files:
            lines = self.texts[name].splitlines()
            headings += [m.group(2) for m in map(HEADING.match, lines) if m]
            start, level = None, 0
            for i, line in enumerate(lines):
                m = HEADING.match(line)
                if m and section.lower() in m.group(2).lower():
                    start, level = i, len(m.group(1))
                    break
            if start is None:
                continue
            end = len(lines)
            for j in range(start + 1, len(lines)):
                m = HEADING.match(lines[j])
                if m and len(m.group(1)) <= level:
                    end = j
                    break
            results.append(f"[{name}]\n" + "\n".join(lines[start:end]).strip()[:4000])

        if results:
            return "\n\n".join(results)
        return (
            f"No section matching '{section}' for {drug}. "
            f"Available sections: {'; '.join(headings)}"
        )

    def glossary(self, term: str) -> str:
        name = next((n for n in self.texts if n.lower().startswith("glossary")), None)
        if not name:
            return "No glossary file available."
        matches = [
            line.strip()
            for line in self.texts[name].splitlines()
            if term.lower() in line.lower() and line.strip()
        ]
        if not matches:
            return f"No glossary entry found for '{term}'."
        return f"[{name}]\n" + "\n".join(matches[:10])