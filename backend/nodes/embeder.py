"""embed_signature_node
~~~~~~~~~~~~~~~~~~~~~~~
A radically simplified version of the former *EmbederNode*.

* Drops all language‑specific parsing – **no regex, no import harvesting**.
* The *signature* for every text (draft or raw doc) is just the **first
  `MAX_CHARS` characters** of the file.
* Embeds the signatures with **`text-embedding-3-small`**.
* Computes cosine similarity between the draft vector and every raw‑doc vector.
* Mutates `state["raw_docs"]` and `state["initial_code"]` in‑place.

Public API remains identical, so you can swap the file and rerun your pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from openai import AsyncOpenAI

from ..base_node import BaseNode
from ..state import InitialCode, RawDoc

_LOG = logging.getLogger("nodes.embed_signature")
_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
#                         configuration constants
# ---------------------------------------------------------------------------
MAX_CHARS = 10_000        # char slice per document → ~8 k tokens max

def _signature(text: str) -> str:
    """Return the leading slice that goes to the embedder."""
    return text[:MAX_CHARS]

# ---------------------------------------------------------------------------
#                                main node
# ---------------------------------------------------------------------------
class EmbederNode(BaseNode):
    """One‑shot signature embedder using a fixed char‑slice."""

    def __init__(self, client: AsyncOpenAI) -> None:  # noqa: D401
        super().__init__("embed_signature")
        self._client = client

    # ---------------------------------------------------------------------
    #                           pipeline entry point
    # ---------------------------------------------------------------------
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: WPS110
        raw_docs: List[RawDoc] = list(state.get("raw_docs", []))
        draft: InitialCode | None = state.get("initial_code")

        if not raw_docs and draft is None:
            _LOG.warning("EmbederNode: nothing to embed – skipping.")
            return {}

        # ---------------- collect signatures ------------------- #
        signatures: List[str] = []
        for doc in raw_docs:
            sig = _signature(doc["content"])
            doc["signature_text"] = sig
            signatures.append(sig)

        if draft is not None:
            draft_sig = _signature(draft["content"])
            draft["signature_text"] = draft_sig
            signatures.append(draft_sig)
        else:
            draft_sig = None

        # ---------------- call embedding API ------------------- #
        resp = await self._client.embeddings.create(model=_MODEL, input=signatures)
        vectors = [item.embedding for item in resp.data]

        if draft_sig is not None:
            draft_vec = np.asarray(vectors.pop(), dtype=np.float32)
            draft["embedding"] = draft_vec
        else:
            draft_vec = None

        # ------------- cosine similarity & state update -------- #
        for doc, vec in zip(raw_docs, vectors, strict=False):
            arr = np.asarray(vec, dtype=np.float32)
            doc["embedding"] = arr
            if draft_vec is not None:
                sim = float(np.dot(draft_vec, arr) / (np.linalg.norm(draft_vec) * np.linalg.norm(arr)))
            else:
                sim = None
            doc["similarity_score"] = sim

        _LOG.info("EmbederNode: embedded %d raw docs%s.", len(raw_docs), " + draft" if draft else "")
        return {"raw_docs": raw_docs, "initial_code": draft}
