import logging
import httpx
from app.config import N8N_WEBHOOK_URL

logger = logging.getLogger("pharma-agent")


def assess_risk(review: str) -> str:
    """'high' if the critic asked for a revision, else 'low'."""
    return "high" if "VERDICT: REVISE" in review.upper() else "low"


def notify_n8n(payload: dict) -> None:
    """Send the event to n8n. Never raises: a failure must not break /ask."""
    if not N8N_WEBHOOK_URL:
        return
    try:
        httpx.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
    except httpx.HTTPError as e:
        logger.warning("n8n notification failed: %s", e)