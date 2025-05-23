"""
    PlannerNode turns the latest user message into a structured outline for the drafter node
    and creates 3 or less Tavily search queries for the search node.
"""

# planner.py
import logging, textwrap
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

from ..base_node   import BaseNode
from ..llm_configs import LLMConfig

_log = logging.getLogger("backend.nodes.planner")


# structure of LLM reply
class _Out(BaseModel):
    solution_outline: str
    search_queries:   List[str] = Field(max_items=3)


# ---------- planner node ----------
class PlannerNode(BaseNode):
    
    # init node and log graph transitions
    def __init__(self, llm: AsyncOpenAI):
        super().__init__("planner")
        self.llm = llm
        self.cfg = LLMConfig.PLANNER

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        # validate user message 
        try:
            last_msg = state["messages"][-1]
        except (KeyError, IndexError):
            raise ValueError("PlannerNode: state.messages is empty")

        if not isinstance(last_msg, HumanMessage):
            raise TypeError(
                f"PlannerNode: expected last message to be HumanMessage, got {type(last_msg).__name__}"
            )

        user_problem = last_msg.content.strip()
        if not user_problem:
            raise ValueError("PlannerNode: user problem text is empty")

        # messages for the LLM
        system_prompt = self.cfg.prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_problem},
        ]

        # call llm
        resp = await self.llm.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
        )

        raw_json = textwrap.dedent(resp.choices[0].message.content).strip()
        parsed   = _Out.model_validate_json(raw_json)
    
        
        # print results
        print("\nPlanner:")
        print(f"Outline:\n{parsed.solution_outline.strip()}\n")
        print(f"Tavily queries ({len(parsed.search_queries)}):")
        for query in parsed.search_queries:
            print(f" • {query}")

        # log results
        _log.info(
            "\n\n----- Planner output -----\nOutline:\n%s\n\nTavily queries (%d):\n%s",
            parsed.solution_outline.strip(),
            len(parsed.search_queries),
            "\n".join(f" • {q}" for q in parsed.search_queries),
        )
        
        # update state
        return {
            "solution_outline": parsed.solution_outline,
            "search_queries":   parsed.search_queries,
            "messages":         [AIMessage(content=raw_json)],
        }
