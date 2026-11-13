import difflib
import logging
from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger("pharma-agent")

# NOTE: literal JSON braces are doubled ({{ }}) because this is a ChatPromptTemplate.
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a RAG Faithfulness Evaluator.
        Determine if the answer is 100 percent grounded in the retrieved Markdown context.
        Respond ONLY with a JSON object, no other text:
        {{"faithfulness_score": <float between 0.0 and 1.0>, "ungrounded_claims": [<strings>]}}"""
    ),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])

PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Context Precision Evaluator.
        Calculate the ratio of retrieved Markdown passages that were relevant to answering the user question.
        Respond ONLY with a JSON object, no other text:
        {{"context_precision": <float between 0.0 and 1.0>, "reason": "<short explanation>"}}"""
    ),
    ("human", "Question:\n{question}\n\nRetrieved Context:\n{context}")
])


class RAGEvaluator:
    """Evaluates RAG execution using Faithfulness, Context Precision, and String Similarity."""

    def __init__(self):
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        parser = JsonOutputParser()
        self.faithfulness_chain = FAITHFULNESS_PROMPT | llm | parser
        self.precision_chain = PRECISION_PROMPT | llm | parser

    def evaluate_string_similarity(self, prediction: str, reference: str) -> float:
        """Computes SequenceMatcher string similarity score."""
        return round(difflib.SequenceMatcher(None, prediction.strip(), reference.strip()).ratio(), 4)

    @staticmethod
    def _safe(chain, inputs: dict) -> Dict[str, Any]:
        """A judge failure (bad JSON, API error) must not break the endpoint."""
        try:
            return chain.invoke(inputs)
        except Exception as e:
            logger.warning("Evaluation chain failed: %r", e)
            return {"error": str(e)}

    def evaluate_rag_triplet(self, question: str, context: str, answer: str, reference: str = None) -> Dict[str, Any]:
        """Runs evaluation across Context Precision, Faithfulness, and String Similarity metrics."""
        faithfulness_res = self._safe(self.faithfulness_chain, {"context": context, "answer": answer})
        precision_res = self._safe(self.precision_chain, {"question": question, "context": context})
        string_score = self.evaluate_string_similarity(answer, reference) if reference else None

        return {
            "context_precision": precision_res,
            "faithfulness": faithfulness_res,
            "string_similarity": string_score
        }