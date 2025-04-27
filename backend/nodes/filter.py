"""
    Filter node: chooses up to three URLs worth crawling.
    The URLs are placed directly in state['crawl_urls']; 
"""

# filter.py
from __future__ import annotations
import logging, textwrap
from string import Template
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

from ..base_node   import BaseNode
from ..llm_configs import LLMConfig
from ..state       import SearchDoc

_log = logging.getLogger("backend.nodes.filter")


# structure for LLM reply
class _Out(BaseModel):
    selected_urls: List[str] = Field(min_items=1, max_items=3)


# -------------- filter node  ----------------
class FilterNode(BaseNode):
    def __init__(self, llm: AsyncOpenAI):
        super().__init__("filter")
        self.llm = llm
        self.cfg = LLMConfig.FILTER

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # get context from state
        user_problem = state["messages"][0].content
        outline      = state["solution_outline"]
        docs         = state["search_docs"]

        if not docs:
            raise ValueError("FilterNode: search_docs is empty")

        # build numbered url title and score list for the llm to choose from
        def _fmt(d: SearchDoc, i: int) -> str:
            score = f"{d.get('score', 0):.2f}" if d.get("score") is not None else "—"
            return f"[{i:02}] score={score}  {d['url']}  {d['title'] or ''}"

        docs_block = "\n".join(_fmt(d, i) for i, d in enumerate(docs, 1))

        # chat messages
        system_msg = self.cfg.prompt  
        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": user_problem},
            {"role": "assistant", "content": f"Solution outline:\n{outline}"},
            {"role": "user",      "content": docs_block},
        ]

        resp = await self.llm.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
        )
        raw_reply = resp.choices[0].message.content
        _log.info("FilterNode raw reply: %s", raw_reply)

        # validate JSON
        try:
            parsed = _Out.model_validate_json(raw_reply)
        except ValidationError as exc:
            _log.error("FilterNode: invalid JSON → %s", exc)
            raise

        # update state
        return {
            "crawl_urls": parsed.selected_urls,
            "messages":   [AIMessage(content=raw_reply)],
        }
