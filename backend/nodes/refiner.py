from __future__ import annotations

import logging, textwrap
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

from ..base_node   import BaseNode
from ..state       import RawDoc
from ..llm_configs import LLMConfig

_log = logging.getLogger("backend.nodes.refiner")

# ───────── limits ───────── #
_TOP_K     = 3
_MAX_CHARS = 6_000
# ────────────────────────── #


class _Refined(BaseModel):
    content: str = Field(..., description="Polished runnable code.")


def _best(docs: List[RawDoc], k: int) -> List[RawDoc]:
    """Return the top-k docs by similarity score (or the first k if un-scored)."""
    scored = [d for d in docs if d.get("similarity_score") is not None]
    scored.sort(key=lambda d: d["similarity_score"], reverse=True)
    return scored[:k] if scored else docs[:k]


def _clip(txt: str, limit: int = _MAX_CHARS) -> str:
    """Hard-truncate long example bodies so the context stays within limits."""
    return txt if len(txt) <= limit else txt[: limit - 30] + " …"


class RefinerNode(BaseNode):
    """Second-stage node that upgrades the initial draft using retrieved examples."""

    def __init__(self, llm: AsyncOpenAI):
        super().__init__("refiner")
        self.llm = llm
        self.cfg = LLMConfig.REFINER

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # ── read state ──────────────────────────────────────────────────────
        problem: str = state["messages"][0].content.strip()
        draft: str   = state["initial_code"]["content"].strip()
        raw_docs: List[RawDoc] = state.get("raw_docs", [])
        top_docs = _best(raw_docs, _TOP_K)

        _log.info("Refiner: %d raw docs → top %d", len(raw_docs), len(top_docs))

        # ── build examples block ───────────────────────────────────────────
        examples_block = "\n\n".join(
            f"[{i+1}] {d['url']}\n{_clip(d['content'])}" for i, d in enumerate(top_docs)
        ) or "(no examples)"

        # ── compose chat messages ──────────────────────────────────────────
        messages = [
            {"role": "system",    "content": self.cfg.prompt},                       # refinement rules
            {"role": "user",      "content": problem},                               # original problem
            {
                "role": "assistant",
                "content": f"Current draft code:\n```python\n{draft}\n```",
            },
            {
                "role": "assistant",
                "content": (
                    "Reference examples (full text below). "
                    "Improve the draft; cite a URL above reused blocks.\n\n"
                    f"{examples_block}"
                ),
            },
        ]

        _log.debug("Refiner: prompt size %d chars",
                   sum(len(m["content"]) for m in messages))

        # ── call LLM ───────────────────────────────────────────────────────
        resp = await self.llm.chat.completions.create(
            model       = self.cfg.model,
            temperature = self.cfg.temperature,
            messages    = messages,
        )

        raw_code = textwrap.dedent(resp.choices[0].message.content).strip()

        # ── validate output ────────────────────────────────────────────────
        try:
            refined = _Refined(content=raw_code)
        except ValidationError as exc:
            _log.error("Refiner: invalid LLM content %s", exc)
            raise

        _log.info("Refiner: produced %d chars", len(refined.content))

        # ── state diff ─────────────────────────────────────────────────────
        return {
            "final_content": {
                "content":    refined.content,
                "sources":    [d["url"] for d in top_docs],
                "reflection": None,
            },
            "messages": [AIMessage(content=refined.content)],
            "status":   "refined",
        }
