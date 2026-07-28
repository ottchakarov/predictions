"""L2 - Agentic RAG layer: unstructured-text signals for the L4 models.

Turns messy text (injury reports, team news, sentiment) into a single bounded
squad-state score in [-1, 1] that the Dixon-Coles model fuses into team strengths.
Every retrieval is point-in-time (``date < as_of``) to keep backtests leakage-free
(data_hygiene.mdc).
"""
from .sentiment import NewsFetcher, NewsItem, SentimentAgent

__all__ = ["NewsFetcher", "NewsItem", "SentimentAgent"]
