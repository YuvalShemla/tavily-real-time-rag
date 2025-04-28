"""
    ResponderNode: Prints a concise run summary (doc counts, final code, similarity sorted sources).
    Prompts the user for further instructions and expects a valid JSON:
    {"status": "continue"|"done", "problem"?: "…", "goodbye"?: "…"}`.
    updates state accordingly, and wipes all previous keys so a new planning cycle starts clean.
"""

#responder.py
from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

from ..base_node import BaseNode
from ..llm_configs import LLMConfig


_log = logging.getLogger("backend.nodes.follow_up")

# structure of the LLM reply
class _Out(BaseModel):
    status: Literal["continue", "done"]
    problem: Optional[str] = Field(None, description="Rewritten task if we loop")
    goodbye: Optional[str] = Field(None, description="Optional farewell text")

    # strict validation for loop decision
    @classmethod
    def validate(cls, value):  # type: ignore[override]
        obj = super().model_validate(value)
        if obj.status == "continue" and not (obj.problem and obj.problem.strip()):
            raise ValueError("problem required when status='continue'")
        return obj


# helper functions
def _print_run_summary(state: Dict[str, Any]) -> None:
    """print and log a quick recap of the latest run."""
    lines: list[str] = []

    # header
    lines.append("Run summary (latest cycle)")

    # counts
    search_count = len(state.get("search_docs", []))
    crawl_count  = len(state.get("crawl_docs",  []))
    extract_count= len(state.get("raw_docs",   []))
    lines.append(f"Tavily Search docs:   {search_count}")
    lines.append(f"Tavily Crawl docs:    {crawl_count}")
    lines.append(f"Tavily Extract docs:  {extract_count}")

    # final code
    final_code = state.get("final_content", {}).get("content")
    if final_code:
        lines.append("")  # blank line
        lines.append("Final Code:")
        lines.append(final_code)
        lines.append("=" * 80)

    # similarity‐sorted sources
    raw_docs = [
        d for d in state.get("raw_docs", [])
        if d.get("similarity_score") is not None
    ]
    raw_docs.sort(key=lambda d: d["similarity_score"], reverse=True)

    if raw_docs:
        lines.append("")  # blank line
        lines.append("Raw documents by similarity:")
        for d in raw_docs:
            lines.append(f"{d['url']}: {d['similarity_score']:.4f}")
    else:
        lines.append("")  # blank line
        lines.append("No similarity scored raw docs.")

    # print to console
    for line in lines:
        print(line)

    # log at INFO
    _log.info("\n".join(lines))

# clean the state before planner
def _clean_state(state: Dict[str, Any]) -> None:
    """Remove keys that shouldn't carry over into the next planning cycle."""
    for k in (
        # planning
        "solution_outline",
        "search_queries",
        # tavily docs
        "search_docs",
        "crawl_urls",
        "crawl_docs",
        "raw_docs",
        # content
        "initial_content",
        "final_content",
    ):
        state.pop(k, None)

# ---------- Responder node -----------
class ResponderNode(BaseNode):
    def __init__(self, llm: AsyncOpenAI):
        super().__init__("responder")
        self.llm = llm
        self.cfg = LLMConfig.Responder
        self._default_goodbye = "Glad I could help — good luck!"

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:  # noqa: WPS110

        # output summary
        _print_run_summary(state)

        # prompt user for follow ups 
        user_input = input(" Anything else I can help with?\n> ").strip()
        if not user_input:
            _clean_state(state)
            bye_msg = AIMessage(content=self._default_goodbye)
            return {"status": "done", "messages": [bye_msg]}
        
        final_content = state.get("final_content", {}).get("content", "").strip()
        code_snippet = final_content[:3000] + (" …" if len(final_content) > 3000 else "")

        # prepare message list for the follow-up LLM
        messages = [
            {"role": "system", "content": self.cfg.prompt},
            {"role": "assistant",
                "content": f"Here is the code that was previously produced:\n```python\n{code_snippet}\n```"},
            {"role": "user",   "content": user_input},
        ]
        resp = await self.llm.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
        )
        raw_json = textwrap.dedent(resp.choices[0].message.content).strip()

        # validate response
        try:
            payload = _Out.model_validate_json(raw_json)
        except (json.JSONDecodeError, ValidationError) as err:
            _log.error("ResponderNode: invalid JSON - %s", err)
            _clean_state(state)
            err_msg = AIMessage(content=f"[ResponderNode error] {err}")
            return {"status": "done", "messages": [err_msg]}

        # build the state update 
        updates: Dict[str, Any] = {
            "status": payload.status,
            "follow_up_response": raw_json,
            "messages": [AIMessage(content=raw_json)],
        }

        # Both logs the messege and prints it to the consule
        _log.info(
            "\n\n ----- Responder loop information ----- \n"
            "Printed results to the user, LLM follow-up status=%s\n",
            payload.status,
        )


        
        # if we continue, add the problem to the state
        if payload.status == "continue":
            updates["messages"].append(HumanMessage(content=payload.problem.strip()))
            _clean_state(state)
        else:  # done
            goodbye = payload.goodbye.strip() if payload.goodbye else self._default_goodbye
            print(f"\n {goodbye}\n")
            updates["messages"].append(AIMessage(content=goodbye))
            _clean_state(state)

        # return the updates to the state
        return updates
