"""
Run the LangGraph pipeline (planner → search → drafter → filter).
"""

from __future__ import annotations
import asyncio, logging, os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from openai import AsyncOpenAI
from tavily import TavilyClient

from .state  import State
from .nodes  import (PlannerNode, SearchNode, DrafterNode, FilterNode, CrawlNode,
                        ExtractNode, RankerNode, RefinerNode, ResponderNode)

# ───────────  logging  ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s → %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backend.main")

# load environment variables from .env file
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
if not OPENAI_KEY or not TAVILY_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and TAVILY_API_KEY in .env")

# initialize the Tavily and the LLM clients 
tavily  = TavilyClient(TAVILY_KEY)
llm     = AsyncOpenAI(api_key=OPENAI_KEY)


# ───────────  graph  ───────────
builder = StateGraph(State)
builder.add_node("planner",   PlannerNode(llm))
builder.add_node("search",    SearchNode(tavily))
builder.add_node("drafter",   DrafterNode(llm))
builder.add_node("filter",    FilterNode(llm))
builder.add_node("crawl",     CrawlNode(TAVILY_KEY))
builder.add_node("extract",   ExtractNode(tavily))
builder.add_node("ranker",   RankerNode(llm))
builder.add_node("refiner",   RefinerNode(llm))
builder.add_node("responder", ResponderNode(llm))

builder.set_entry_point("planner")
builder.add_edge("planner", "search")
builder.add_edge("planner", "drafter")
builder.add_edge("search", "filter")
builder.add_edge("filter", "crawl")
builder.add_edge("crawl", "extract")
builder.add_edge("extract", "ranker")
builder.add_edge("ranker", "refiner")
builder.add_edge("refiner", "responder")

# conditional from follow_up 
builder.add_conditional_edges(
    "responder",
    lambda s: END if s.get("status") == "done" else "planner",
    {"planner": "planner", END: END},
)

graph = builder.compile()
log.info("✅  Graph compiled")


# ───────────  run  ───────────
async def main() -> None:
    problem = input("📝  Describe your programming problem:\n> ").strip()
    if not problem:
        print("No input – exiting.")
        return

    init_state: State = {
        "messages": [HumanMessage(content=problem)],
        "status":   "new",
    }

     # recursion_limit for max two loops, might need to make dynamic for additional loops
    final_state: State = await graph.ainvoke(init_state, config={"recursion_limit": 20})

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
