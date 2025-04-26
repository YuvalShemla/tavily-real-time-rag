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
from .utils  import clip
from .nodes  import PlannerNode, SearchNode, DrafterNode, FilterNode, CrawlNode, ExtractNode, EmbederNode

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
builder.add_node("planner",  PlannerNode(llm))
builder.add_node("search",   SearchNode(tavily))
builder.add_node("drafter",  DrafterNode(llm))
builder.add_node("filter",   FilterNode(llm))
builder.add_node("crawl",    CrawlNode(TAVILY_KEY))
builder.add_node("extract",   ExtractNode(tavily))
builder.add_node("embeder", EmbederNode(llm))

builder.set_entry_point("planner")
builder.add_edge("planner", "search")
builder.add_edge("planner", "drafter")
builder.add_edge("search",  "filter")
builder.add_edge("filter",  "crawl")
builder.add_edge("crawl",  "extract")
builder.add_edge("extract",  "embeder")
builder.add_edge("embeder",    END)  
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

    final_state: State = await graph.ainvoke(init_state, config={"recursion_limit": 6})

    # print("\n🔎  Search results")
    # for i, d in enumerate(final_state.get("search_docs", []), 1):
    #     print(f"[{i:02}] {clip(d.get('title'))}  url: {d['url']}")

    print("\n✅  Chosen URLs")
    for url in final_state.get("crawl_urls", []):
        print(f"url: {url}")

    # print("\n✅  Crawled results")
    # for i, d in enumerate(final_state.get("crawl_docs", []), 1):
    #     print(f"[{i:02}] {d.get('url')}  content: {clip(d['content'])}")

    print("\n✅  Extract results")
    for i, d in enumerate(final_state.get("raw_docs", []), 1):
        print(f"[{i:02}] {d.get('url')}  raw_content: {clip(d['content'])}")

    print("🧩  Raw chunks :", len(final_state.get("raw_chunks", [])))
    print("🧩  Code chunks:", len(final_state.get("code_chunks",[])))

    if final_state.get("raw_chunks") and final_state.get("code_chunks"):
        print("✅  Embeddings generated for both raw and draft code.")
    # print("\n💾 Draft code\n")
    # print(final_state["initial_code"]["content"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
