"""
   Refiner node: polishes the initial draft by matching it with the top-K
   most similar raw documents and hopefully producing a better, citation-backed version.
"""

#refiner.py
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

# llm prompt params  
TOP_K_example     = 3  
MAX_CHARS = 8_000  

# helper function:
def top_k_best(docs: List[RawDoc], k: int) -> List[RawDoc]:
    """Sorts the documents by similarity score and returns the top-k."""
    scored = [d for d in docs if d.get("similarity_score") is not None]
    scored.sort(key=lambda d: d["similarity_score"], reverse=True)
    return scored[:k] if scored else docs[:k]

def _clip(txt: str, limit: int = MAX_CHARS) -> str:
    """Hard-truncate long example bodies so the context stays within limits."""
    return txt if len(txt) <= limit else txt[: limit - 30] + " …"

# structure of the LLM reply
class _Refined(BaseModel):
    content: str = Field(..., description="Polished runnable code.")


# -------------- Refiner node  ----------------
class RefinerNode(BaseNode):

    # init node and log graph transitions
    def __init__(self, llm: AsyncOpenAI):
        super().__init__("refiner")
        self.llm = llm
        self.cfg = LLMConfig.REFINER

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        # get context from state
        problem: str = state["messages"][0].content.strip()
        draft: str   = state["initial_content"]["content"].strip()
        raw_docs: List[RawDoc] = state.get("raw_docs", [])
        top_docs = top_k_best(raw_docs, TOP_K_example)

        _log.info("Refiner: %d raw docs → top %d", len(raw_docs), len(top_docs))

        # build the examples block 
        examples_block = "\n\n".join(
            f"[{i+1}] {d['url']}\n{_clip(d['content'])}" for i, d in enumerate(top_docs)
        ) or "(no examples)"

        # create the messages to send to the LLM
        messages = [
            {"role": "system",    "content": self.cfg.prompt},                 # refinement rules
            {"role": "user",      "content": problem},                         # original problem
            {
                "role": "assistant",
                "content": f"Current draft code:\n```python\n{draft}\n```",    # draft content
            },
            {
                "role": "assistant",
                "content": (
                    "Reference examples (full text below). "
                    "Improve the draft; cite a URL above reused blocks.\n\n"
                    f"{examples_block}"                                         # example content
                ),
            },
        ]

        # call LLM 
        resp = await self.llm.chat.completions.create(
            model       = self.cfg.model,
            temperature = self.cfg.temperature,
            messages    = messages,
        )

        # parse LLM reply 
        raw_code = textwrap.dedent(resp.choices[0].message.content).strip()

        # validate output
        try:
            refined = _Refined(content=raw_code)
        except ValidationError as exc:
            _log.error("Refiner: invalid LLM content %s", exc)
            raise

        # log and update state
        print(f"\nRefiner: produced {len(refined.content)} chars of code.")
        _log.info("\n\n ------ Refiner output (500\%s chars) ----- \n%s",
                  len(refined.content),
                  refined.content[:500] + (" …" if len(refined.content) > 500 else ""))
        
        # update state
        return {
            "final_content": {
                "content":    refined.content,
                "sources":    [d["url"] for d in top_docs],
                "reflection": None,
            },
            "messages": [AIMessage(content=refined.content)],
            "status":   "refined",
        }
