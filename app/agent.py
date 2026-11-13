import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from app.config import (
    CHAT_MODEL,
    DISCLAIMER,
    MAX_EVIDENCE_CHARS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RECURSION_LIMIT,
)
from app.rag import KnowledgeBase
from app.graph_rag import GraphRAGEngine

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a planning assistant for a pharmacology research agent. 
            Decompose the question delimited by <question> tags into 2 to 5 short, 
            specific sub-questions or steps. Name the drug(s) in each step. 
            Output a numbered list only, with no extra text.""",
        ),
        ("human", "<question>{question}</question>"),
    ]
)

AGENT_PROMPT = (
    """You are a pharmacology research assistant. You receive a question and a plan.
    Follow the plan step by step. Use the tools to collect evidence from Markdown notices:
    Rules:
    - Use ONLY information returned by the tools (Sparse BM25, Dense Vector, or Graph RAG). Never use outside knowledge.
    - If the Markdown documents do not contain the answer, say so clearly.
    - For comparisons, retrieve evidence for each drug separately.
    - Cite the Markdown source file name for every claim, e.g. (dailymed-warfarin.md).
    - Write a concise draft answer."""
)

CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict reviewer. Check a draft answer against the evidence. 
            Only the evidence delimited by <evidence> tags counts as support.
            Output exactly this format:
            VERDICT: PASS or REVISE
            UNSUPPORTED CLAIMS: a list of claims not backed by the evidence, or 'none'
            CLINICALLY AMBIGUOUS CLAIMS: a list of claims that are vague, conditional or could be misread clinically, or 'none'
            MISSING: important points from the evidence that the draft omitted, or 'none'""",
        ),
        (
            "human",
            """<question>{question}</question>
                <draft>{draft}</draft>
                <evidence>{evidence}</evidence>""",
        ),
    ]
)

FINAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You write the final answer to a pharmacology question for a student.
            Start from the draft and apply the review: remove unsupported claims, add the missing points, and mark ambiguous claims with a short caveat. 
            Use only the evidence. Cite source Markdown file names in parentheses after each claim, e.g. (dailymed-warfarin.md). Use short paragraphs or bullet points. 
            Do not add the disclaimer, it is added automatically.""",
        ),
        (
            "human",
            """<question>{question}</question>
            <draft>{draft}</draft>
            <review>{review}</review>
            <evidence>{evidence}</evidence>""",
        ),
    ]
)


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


class PharmaAgent:
    def __init__(self, kb: KnowledgeBase, graph_engine: GraphRAGEngine):
        llm = get_llm()
        parser = StrOutputParser()
        
        @tool
        def search_hybrid(query: str) -> str:
            """Hybrid Sparse BM25 + Dense Vector search over Markdown drug documents.
            Use it for contraindications, dosages, and general queries."""
            return kb.search_hybrid(query)

        @tool
        def search_bm25_exact(query: str) -> str:
            """Sparse BM25 search for exact medical keyword and dosage matching in Markdown files."""
            return kb.search_bm25(query)

        @tool
        def query_knowledge_graph(entity: str) -> str:
            """Query relational Knowledge Graph for structured drug contraindications."""
            return graph_engine.query_graph_relations(entity)

        @tool
        def get_drug_section(drug: str, section: str) -> str:
            """Return a full section of one drug's Markdown document."""
            return kb.get_section(drug, section)

        @tool
        def lookup_glossary(term: str) -> str:
            """Look up a medical term in the Markdown glossary."""
            return kb.glossary(term)

        @tool
        def list_available_drugs() -> str:
            """List the available Markdown drug files in the corpus."""
            return kb.list_drugs()

        tools = [
            search_hybrid,
            search_bm25_exact,
            query_knowledge_graph,
            get_drug_section,
            lookup_glossary,
            list_available_drugs,
        ]

        self.planner = PLANNER_PROMPT | llm | parser
        self.agent = create_agent(llm, tools, prompt=AGENT_PROMPT)
        self.critic = CRITIC_PROMPT | llm | parser
        self.finalizer = FINAL_PROMPT | llm | parser

    def ask(self, question: str) -> dict:
        plan = self.planner.invoke({"question": question})

        result = self.agent.invoke(
            {"messages": [("human", f"Question: {question}\n\nPlan:\n{plan}")]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        messages = result["messages"]
        draft = messages[-1].content
        evidence = "\n\n".join(m.content for m in messages if m.type == "tool")
        evidence = evidence[:MAX_EVIDENCE_CHARS]
        
        sources = sorted(set(re.findall(r"\[([^\]\n]+\.md)\]", evidence)))
        
        review = self.critic.invoke(
            {"question": question, "draft": draft, "evidence": evidence}
        )

        answer = self.finalizer.invoke(
            {"question": question, "draft": draft, "review": review, "evidence": evidence}
        )
        answer = f"{answer.strip()}\n\n{DISCLAIMER}"

        return {
            "answer": answer,
            "sources": sources,
            "plan": plan,
            "review": review,
            "evidence": evidence,
        }