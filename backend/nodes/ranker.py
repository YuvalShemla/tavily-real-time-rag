"""
    RankerNode embeds the first MAX_CHARS of a file with text-embedding-3-small
    Computes cosine similarity between the draft vector and every rawdoc vector.
    Updates ranking to state["raw_docs"] and state["initial_code"].
"""

# ranker.py
from __future__ import annotations
import logging
from typing import Any, Dict, List

import numpy as np
from openai import AsyncOpenAI

from ..base_node import BaseNode
from ..state import InitialContent, RawDoc


_LOG = logging.getLogger("nodes.ranker")
_MODEL = "text-embedding-3-small"

# Number of first characters to slice for embedding
MAX_CHARS = 8_000

# -------- Ranker Node ---------
class RankerNode(BaseNode):

    # init node and log graph transitions
    def __init__(self, client: AsyncOpenAI) -> None:  # noqa: D401
        super().__init__("ranker")
        self._client = client

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: WPS110
        raw_docs: List[RawDoc] = list(state.get("raw_docs", []))
        draft: InitialContent | None = state.get("initial_content")

        if not raw_docs and draft is None:
            _LOG.warning("EmbederNode: nothing to embed - skipping.")
            return {}

        # Collect signatures for raw docs and draft (improve logic for better results)
        def _signature(text: str) -> str:
            return text[:MAX_CHARS]

        signatures: List[str] = []
        for doc in raw_docs:
            sig = _signature(doc["content"])
            doc["signature_text"] = sig
            signatures.append(sig)

        if draft is not None:
            draft_sig = _signature(draft["content"])
            draft["signature_text"] = draft_sig
            signatures.append(draft_sig) # draft is the last item
        else:
            draft_sig = None

        # request the embeddings  
        resp = await self._client.embeddings.create(model=_MODEL, input=signatures)
        vectors = [item.embedding for item in resp.data]

        # add embedding to the draft and pop it from the list
        if draft_sig is not None:
            draft_vec = np.asarray(vectors.pop(), dtype=np.float32)
            draft["embedding"] = draft_vec
        else:
            draft_vec = None

        #  add embedding to the raw docs and compute similarity to draft
        for doc, vec in zip(raw_docs, vectors, strict=False):
            arr = np.asarray(vec, dtype=np.float32)
            doc["embedding"] = arr
            if draft_vec is not None:
                sim = float(np.dot(draft_vec, arr) / (np.linalg.norm(draft_vec) * np.linalg.norm(arr)))
            else:
                sim = None
            doc["similarity_score"] = sim

        # prepare ranked list
        ranked = sorted(
            raw_docs,
            key=lambda d: (d["similarity_score"] is None,
                        -d["similarity_score"] if d["similarity_score"] is not None else 0.0),
        )
        lines = "\n".join(f" • {d['similarity_score']:.4f} | {d['url']}" for d in ranked)

        # print results
        print(f"\nEmbederNode:\nEmbedded {len(raw_docs)} raw docs" + (" + draft" if draft else "") + ".")

        print(f"\nEmbederNode ranking ({len(ranked)} results):\n{lines}")

        # log results
        _LOG.info(
            "EmbederNode: embedded %d raw docs%s.",
            len(raw_docs),
            " + draft" if draft else "",
        )
        _LOG.info(
            "\n\n----- EmbederNode ranking (%d results): -----\n%s",
            len(ranked),
            lines,
        )

        # update state
        return {"raw_docs": raw_docs, "initial_content": draft}
