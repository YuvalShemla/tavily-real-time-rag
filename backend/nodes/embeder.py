"""
EmbederNode (OpenAI)
• Splits each file into ~1 500-char chunks (300-char overlap)
• Calls OpenAI text-embedding-3-small in batches
• Stores raw_chunks / code_chunks with ready-to-use vectors
"""

from __future__ import annotations
import asyncio, logging, uuid
from typing import Any, Dict, List, Sequence

from openai import AsyncOpenAI
from ..base_node import BaseNode
from ..state     import RawDoc, InitialCode

_log = logging.getLogger("nodes.embed_oa")


class EmbederNode(BaseNode):
    CHUNK_CHARS   = 1_500
    OVERLAP_CHARS = 300
    BATCH_SIZE    = 96               # ≤ 8191 tokens combined per request
    MODEL         = "text-embedding-3-small"

    def __init__(self, client: AsyncOpenAI):
        super().__init__("embed")
        self.client = client

    # ── helpers ──────────────────────────────────────────────────── #
    def _split(self, text: str, url: str | None) -> List[Dict[str, Any]]:
        step = self.CHUNK_CHARS - self.OVERLAP_CHARS
        out  = []
        for i in range(0, len(text), step):
            chunk = text[i : i + self.CHUNK_CHARS]
            out.append(
                dict(
                    id      = uuid.uuid4().hex[:8],
                    url     = url,
                    content = chunk,
                    embedding = None,
                )
            )
            if i + self.CHUNK_CHARS >= len(text):
                break
        return out

    async def _embed_batch(self, batch: Sequence[Dict[str, Any]]) -> None:
        strings = [c["content"] for c in batch]
        resp = await self.client.embeddings.create(
            model=self.MODEL,
            input=strings,
        )
        for c, item in zip(batch, resp.data):
            c["embedding"] = item.embedding  # already list[float]

    # ── graph step ───────────────────────────────────────────────── #
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_docs: List[RawDoc]        = state.get("raw_docs", [])
        draft:     InitialCode | None = state.get("initial_code")

        if not raw_docs and not draft:
            _log.warning("OpenAI Embeder: nothing to embed.")
            return {}

        raw_chunks  = [ch for doc in raw_docs for ch in self._split(doc["content"], doc["url"])]
        code_chunks = self._split(draft["content"], None) if draft else []

        # batch in groups of ≤BATCH_SIZE
        for i in range(0, len(raw_chunks + code_chunks), self.BATCH_SIZE):
            chunk_slice = (raw_chunks + code_chunks)[i : i + self.BATCH_SIZE]
            await self._embed_batch(chunk_slice)

        _log.info("OpenAI Embeder: %d raw + %d code chunks embedded.", len(raw_chunks), len(code_chunks))
        return {"raw_chunks": raw_chunks, "code_chunks": code_chunks}
