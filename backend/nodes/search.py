""" 
    SearchNode: runs Tavily queries and returns a list of the results.
    results dict include: title, url, content, score (relevance).
"""

# search.py
from __future__ import annotations
import asyncio, logging
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError
from tavily import TavilyClient

from ..base_node import BaseNode
from ..state     import SearchDoc

_log = logging.getLogger("backend.nodes.search")


#single tavily search result structure
class TavilyDoc(BaseModel):
    title:   str | None = None
    url:     str
    content: str | None = None
    score:   float | None = None

    model_config = dict(extra="ignore")

# node response structure
class TavilyResp(BaseModel):
    results: List[TavilyDoc]


# ---------- search node  ----------
class SearchNode(BaseNode):

    # init node and log graph transitions
    def __init__(self, client: TavilyClient):
        super().__init__("search")
        self.client = client

    # executes a single tavily search call
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
            parsed = TavilyResp.model_validate(raw)

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

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        queries: List[str] = state.get("search_queries", [])
        if not queries:
            _log.warning("SearchNode: no queries.")
            return {}

        # run tavily queries in parallel
        nested = await asyncio.gather(*[self._run_one(q) for q in queries])
        docs: List[SearchDoc] = [doc for sub in nested for doc in sub]
        
        # log and update state
        _log.info("SearchNode: gathered %d docs from %d queries", len(docs), len(queries))
        return {"search_docs": docs}
