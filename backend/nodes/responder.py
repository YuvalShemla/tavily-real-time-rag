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
    """print a quick recap of the latest run."""
    print("\nRun summary (latest cycle)")
    # counts
    print(
        f"Tavily Search docs:   {len(state.get('search_docs', []))}\n"
        f"Tavily Crawl docs:    {len(state.get('crawl_docs',  []))}\n"
        f"Tavily Extract docs: {len(state.get('raw_docs',   []))}"
    )

    # output final code
    final_code = state.get("final_content", {}).get("content")
    if final_code:
        print("\n Final Code:\n")
        print(final_code)
        print("\n" + "=" * 80 + "\n")

    # list the source docs by similarity
    raw_docs = [d for d in state.get("raw_docs", []) if d.get("similarity_score") is not None]
    raw_docs.sort(key=lambda d: d["similarity_score"], reverse=True)

    if raw_docs:
        print("noRaw documents by similarity:\n")
        for d in raw_docs:
            print(f"{d['url']}: {d['similarity_score']:.4f}")
    else:
        print("No similarity scored raw docs.")

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
        super().__init__("follow_up")
        self.llm = llm
        self.cfg = LLMConfig.FOLLOW_UP
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
            _log.error("FollowUpNode: invalid JSON – %s", err)
            _clean_state(state)
            err_msg = AIMessage(content=f"[follow‑up error] {err}")
            return {"status": "done", "messages": [err_msg]}

        # build the state update 
        updates: Dict[str, Any] = {
            "status": payload.status,
            "follow_up_response": raw_json,
            "messages": [AIMessage(content=raw_json)],
        }

        _log.info(
            "\n\n ----- Responder information ----- \n"
            "Printed results to the user, LLM follow-up status=%s\nLLM response (first 300 chars): %s",
            payload.status,
            raw_json[:300] + (" …" if len(raw_json) > 300 else ""),
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
