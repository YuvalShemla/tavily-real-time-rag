"""
TypedDict definitions for the LangGraph workflow state.
"""

from typing import Annotated, List, Dict
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import AnyMessage, add_messages


# ───────────────────────── existing docs ───────────────────────── #

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
    signature_text: NotRequired[str]             # e.g. "fmt io math os"
    signature_words: NotRequired[List[str]]      # e.g. ["fmt","io","math","os"]
    embedding: NotRequired[List[float]]          # the vector you sent/received
    similarity_score: NotRequired[float | None]  # cosine vs. draft

class InitialContent(TypedDict):
    content: str
    chunk_ids: List[str] | None
    signature_text: NotRequired[str]             # draft’s import-signature
    signature_words: NotRequired[List[str]]      # list of draft’s modules
    embedding: NotRequired[List[float]]          # draft’s embedding vector

class FinalCode(TypedDict):
    content: str
    sources: List[str] | None
    reflection: str | None



# ───────────────────────────── state ───────────────────────────── #

class State(TypedDict):
    # LangGraph message buffer
    messages: Annotated[List[AnyMessage], add_messages]

    # planning
    solution_outline: NotRequired[str]
    search_queries:   NotRequired[List[str]]

    # web search & crawl
    search_docs: NotRequired[List[SearchDoc]]
    crawl_urls:  NotRequired[List[str]]
    crawl_docs:  NotRequired[List[CrawlDoc]]
    raw_docs:    NotRequired[List[RawDoc]]

    # code generation
    initial_content:  NotRequired[InitialContent]
    final_content: NotRequired[FinalCode]

    # workflow meta
    follow_up_response: NotRequired[str]   # 
    status:                str            # "new" | "continue" | "done"
