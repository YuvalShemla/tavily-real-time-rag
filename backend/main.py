"""
    Entrypoint — build the LangGraph workflow.
    Prompts the user for a problem at each loop iteration.
"""

# main.py
from __future__ import annotations
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from openai import AsyncOpenAI
from tavily import TavilyClient

from .state  import State
from .nodes  import (PlannerNode, SearchNode, DrafterNode, FilterNode, CrawlNode,
                        ExtractNode, RankerNode, RefinerNode, ResponderNode)

from .utils import get_logger, get_keys 

# set up logging
log = get_logger("backend.main")
log.info("\n\n ---------- Starting New Workflow  ---------- \n")

# initialize the Tavily OpenAI clients 
OPENAI_KEY, TAVILY_KEY = get_keys()
tavily  = TavilyClient(TAVILY_KEY)
llm     = AsyncOpenAI(api_key=OPENAI_KEY)

# init state graph and add our nodes
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

# set up edges between the nodes 
builder.set_entry_point("planner")
builder.add_edge("planner", "search")
builder.add_edge("planner", "drafter")
builder.add_edge("search", "filter")
builder.add_edge("filter", "crawl")
builder.add_edge("crawl", "extract")
builder.add_edge("extract", "ranker")
builder.add_edge("ranker", "refiner")
builder.add_edge("refiner", "responder")

# conditional edge from responder 
builder.add_conditional_edges(
    "responder",
    lambda s: END if s.get("status") == "done" else "planner",
    {"planner": "planner", END: END},
)

# compile the graph
graph = builder.compile()
log.info("Graph compiled")


# main runs the entire workflow :D 
async def main() -> None:

    # prompt the user for a problem
    problem = input("Describe your problem:\n> ").strip()
    if not problem:
        print("No input, exiting.")
        return
    
    log.info("\n\nUser problem: %s\n", problem)

    # init the state
    init_state: State = {
        "messages": [HumanMessage(content=problem)],
        "status":   "new",
    }

    # recursion_limit is set for a max of two loops
    # might need to make dynamic for additional loops
    final_state: State = await graph.ainvoke(init_state, config={"recursion_limit": 20}) 

    log.info("Successfully executed the workflow :)\n\n")


# run main
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
