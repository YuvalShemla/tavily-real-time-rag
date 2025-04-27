""" 
    Drafter node — converts the outline and problem text into an initial draft  
    and returns the result to the initial_content in the graph state.
"""

# drafter.py
import logging, textwrap
from typing import Any, Dict

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from ..base_node   import BaseNode
from ..llm_configs import LLMConfig

_log = logging.getLogger("backend.nodes.drafter")

# structure of the llm reply 
class _Code(BaseModel):
    content: str = Field(...)

# ---------- Drafter node ----------
class DrafterNode(BaseNode):

    # init node and log graph transitions
    def __init__(self, llm: AsyncOpenAI):
        super().__init__("drafter")
        self.llm = llm
        self.cfg = LLMConfig.DRAFTER

    # LangGraph entrypoint
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # get context from state 
        user_problem = state["messages"][0].content.strip()
        outline      = state["solution_outline"].strip()

        if not user_problem or not outline:
            raise ValueError("DrafterNode: missing problem or outline")

        # create chat messages for the llm
        messages = [
            {"role": "system",    "content": self.cfg.prompt},                 # drafting rules
            {"role": "user",      "content": user_problem},                    # plain problem
            {"role": "assistant", "content": f"Solution outline:\n{outline}"}, # outline
        ]

        # call the LLM
        resp = await self.llm.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            messages=messages,
        )

        code_text = textwrap.dedent(resp.choices[0].message.content).strip()
        code = _Code(content=code_text)

        # log and update state 
        _log.info("\n\n ----- Draft content: (500\%s chars) ----- \n%s\n",
            len(code_text),
            code_text[:500] + (" … " if len(code_text) > 500 else ""))
        
        return {
            "initial_content": {"content": code.content, "chunk_ids": None},
            "messages":     [AIMessage(content=code.content)],
        }
