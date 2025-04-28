"""
TypedDict definitions for the LangGraph workflow state.
"""

# state.py
from typing import Annotated, List, Dict
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import AnyMessage, add_messages


# classes to store results from the nodes
class SearchDoc(TypedDict):
    title:   str | None
    url:     str | None
    content: NotRequired[str | None]
    score:   float | None

class CrawlDoc(TypedDict):
    url: str
    content: str

class RawDoc(TypedDict):
    url: str
    content: str
    signature_text: NotRequired[str]
    embedding: NotRequired[List[float]]
    similarity_score: NotRequired[float | None]

class InitialContent(TypedDict):
    content: str
    signature_text: NotRequired[str]
    embedding: NotRequired[List[float]]

class FinalCode(TypedDict):
    content: str
    sources: List[str] | None
    reflection: str | None


# ----------- state ----------- 
class State(TypedDict):
    # messages between nodes
    messages: Annotated[List[AnyMessage], add_messages]

    # planner
    solution_outline: NotRequired[str]
    search_queries:   NotRequired[List[str]]

    # tavily search & crawl & extract
    search_docs: NotRequired[List[SearchDoc]]
    crawl_urls:  NotRequired[List[str]]
    crawl_docs:  NotRequired[List[CrawlDoc]]
    raw_docs:    NotRequired[List[RawDoc]]

    # content generation
    initial_content:  NotRequired[InitialContent]
    final_content: NotRequired[FinalCode]

    # workflow 
    follow_up_response: NotRequired[str]
    status:                str
