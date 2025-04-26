"""
Search node — runs Tavily queries concurrently and returns a flat list.
Captures: title, url, content (short description), score (relevance).
"""

from __future__ import annotations
import asyncio, logging
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError
from tavily import TavilyClient

from ..base_node import BaseNode
from ..state     import SearchDoc

_log = logging.getLogger("backend.nodes.search")


#search result structure
class _TDoc(BaseModel):
    title:   str | None = None
    url:     str
    content: str | None = None
    score:   float | None = None

    model_config = dict(extra="ignore")


class _TResp(BaseModel):
    results: List[_TDoc]


# ---------- search node  ----------
class SearchNode(BaseNode):
    """Runs Tavily search queries in parallel and stores List[SearchDoc]."""

    def __init__(self, client: TavilyClient):
        super().__init__("search")
        self.client = client

    async def _run_one(self, query: str) -> List[SearchDoc]:
        _log.info("Tavily query: %s", query)
        try:
            raw: Dict[str, Any] = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth="advanced",
                max_results=6,
                include_domains=["github.com"],
            )
            parsed = _TResp.model_validate(raw)

        except (ValidationError, Exception) as exc:
            _log.error("Tavily error for %r: %s", query, exc)
            return []

        return [
            {
                "title":   d.title,
                "url":     d.url,
                "content": d.content,
                "score":   d.score,
            }
            for d in parsed.results
        ]

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        queries: List[str] = state.get("search_queries", [])
        if not queries:
            _log.warning("SearchNode: no queries.")
            return {}

        nested = await asyncio.gather(*[self._run_one(q) for q in queries])
        docs: List[SearchDoc] = [doc for sub in nested for doc in sub]

        _log.info("SearchNode: gathered %d docs from %d queries", len(docs), len(queries))
        return {"search_docs": docs}
