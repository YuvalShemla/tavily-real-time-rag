"""
Crawler node — calls Tavily /crawl in parallel and stores
List[CrawlDoc]  where each dict has keys  {url, content}.
"""
# TODO change back params
from __future__ import annotations
import asyncio, logging, os, requests
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError
from ..base_node import BaseNode
from ..state     import CrawlDoc

_log = logging.getLogger("backend.nodes.crawler")

# ─────────────────────────────────────────────────────────────────────────────
#   JSON schema that actually matches the /crawl endpoint today
# ─────────────────────────────────────────────────────────────────────────────
class _Page(BaseModel):
    url: str
    raw_content: str | None = None        # full page text

class _CRaw(BaseModel):
    results: List[_Page]                  # << correct top-level key
    model_config = dict(extra="ignore")


# ─────────────────────────────────────────────────────────────────────────────
#   Crawler node
# ─────────────────────────────────────────────────────────────────────────────
class CrawlNode(BaseNode):
    endpoint = "https://api.tavily.com/crawl"
    timeout  = 150                        # seconds

    # Default crawl parameters ------------------------------------------------
    _payload: Dict[str, Any] = dict(
        limit         = 500,
        max_depth     = 3,
        max_breadth   = 100,
        extract_depth = "advanced",
        allow_external= False,
        # query = "code"
        select_paths  = [
            r"/.*\.ipynb$", r"/.*\.py$", r"/.*\.(js|ts|tsx)$",
            r"/.*\.(cpp|c|cc|h|hpp)$", r"/.*\.(go|rs)$", r"/.*\.java$",
            r"/.*\.(md|rst)$", r"/.*\.(yaml|yml|toml|json)$",
        ],
    )

    # --------------------------------------------------------------------- #
    def __init__(self, api_key: str | None = None):
        super().__init__("crawler")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY missing for CrawlerNode")

    # ---------------- helpers ---------------- #
    def _crawl_one_sync(self, base_url: str) -> List[CrawlDoc]:
        """Blocking HTTP call → executed inside a thread."""
        try:
            resp = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={**self._payload, "url": base_url},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            parsed = _CRaw.model_validate(resp.json())

        except (requests.RequestException, ValidationError) as exc:
            _log.error("Tavily crawl failed for %s → %s", base_url, exc)
            return []

        return [
            {"url": page.url, "content": page.raw_content or ""}
            for page in parsed.results
            if page.raw_content                    # skip empty pages
        ]

    async def _crawl_one(self, url: str) -> List[CrawlDoc]:
        return await asyncio.to_thread(self._crawl_one_sync, url)

    # -------------- LangGraph entrypoint -------------- #
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        urls: List[str] = state.get("crawl_urls", [])
        if not urls:
            _log.warning("CrawlerNode: no URLs to crawl.")
            return {}

        nested = await asyncio.gather(*[self._crawl_one(u) for u in urls])
        docs: List[CrawlDoc] = [doc for sub in nested for doc in sub]

        _log.info("CrawlerNode: gathered %d pages from %d base URLs",
                  len(docs), len(urls))
        return {"crawl_docs": docs}
