"""
    Extract node — picks unique GitHub files, converts /blob/→/raw/,
    calls Tavily /extract concurrently (≤20 URLs per batch),
    and returns List[RawDoc] with keys {url, content}.
"""

from __future__ import annotations
import asyncio, logging
from pathlib import PurePosixPath
from typing import Any, Dict, List, Set, Tuple

from tavily import TavilyClient
from ..base_node import BaseNode
from ..state     import CrawlDoc, RawDoc

_log = logging.getLogger("backend.nodes.extract")


# a few short helpers
def blob_to_raw(url: str) -> str:
    """converts GitHub blob URL to raw URL."""
    return url.replace("/blob/", "/raw/", 1) if "/blob/" in url else url

def raw_to_blob(url: str) -> str:
    """converts GitHub raw URL to blob URL."""
    return url.replace("/raw/", "/blob/", 1) if "/raw/" in url else url

def filename(url: str) -> str | None:
    """extracts the filename from a GitHub URL."""
    name = PurePosixPath(url).name
    return name if "." in name else None


# --------- Extract Node ---------
class ExtractNode(BaseNode):
    BATCH_SIZE = 20                # tavily extract requires  ≤20 per request
    PARAMS     = dict(extract_depth="advanced")

    # init node and log graph transitions 
    def __init__(self, client: TavilyClient):
        super().__init__("extract")
        self.client = client

    # executes a single extract call 
    def _extract_sync(self, urls: List[str]) -> Tuple[List[RawDoc], List[Dict[str, str]]]:
        raw_docs: List[RawDoc] = []
        failed:   List[Dict[str, str]] = []

        try:
            resp = self.client.extract(urls=urls, **self.PARAMS)
        except Exception as exc:              # network or quota errors
            _log.error("Tavily extract error for batch: %s", exc)
            return [], [{"url": u, "error": str(exc)} for u in urls]

        for item in resp.get("results", []):
            text = item.get("raw_content")
            if text:
                raw_docs.append({"url": raw_to_blob(item["url"]), "content": text})

        failed.extend(resp.get("failed_results", []))
        return raw_docs, failed
    
    # executes a batch of tavily extract calls
    async def _extract_batch(self, urls: List[str]) -> Tuple[List[RawDoc], List[Dict[str, str]]]:
        return await asyncio.to_thread(self._extract_sync, urls)

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        crawl_docs: List[CrawlDoc] = state.get("crawl_docs", [])
        if not crawl_docs:
            _log.warning("ExtractNode: no crawl_docs provided.")
            return {}

        # keep track of unique filenames and gathers raw URLs to extract
        seen: Set[str] = set()
        raw_docs: List[RawDoc] = []  
        todo: List[str] = []              

        # deduplicate by filename
        for doc in crawl_docs:
            url, text = doc["url"], doc["content"]
            fname = filename(url)
            if not fname or fname in seen:
                continue
            seen.add(fname)

            if "/raw/" in url:
                raw_docs.append({"url": raw_to_blob(url), "content": text})
            elif "/blob/" in url:
                todo.append(blob_to_raw(url))

        _log.info("ExtractNode: %d URLs already had content, %d queued for extract.",
                  len(raw_docs), len(todo))

        # execute extract calls in parallel batches
        tasks = [
            self._extract_batch(todo[i:i + self.BATCH_SIZE])
            for i in range(0, len(todo), self.BATCH_SIZE)
        ]
        batches = await asyncio.gather(*tasks)

        # collect results and log errors
        success = failure = 0
        for docs, fails in batches:
            raw_docs.extend(docs)
            success += len(docs)
            failure += len(fails)
            for f in fails:
                _log.info("Extract failed: %s — %s", f.get("url"), f.get("error"))

        # log results
        _log.info("ExtractNode: extracted %d/%d URLs successfully.", success, success + failure)
        urls_txt = "\n".join(f" • {d['url']}" for d in raw_docs) 
        _log.info("\n\n ----- ExtractNode files (%d total) -----\n%s", len(raw_docs), urls_txt)

        # update state
        return {"raw_docs": raw_docs}
