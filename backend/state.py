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


class InitialCode(TypedDict):
    content: str
    chunk_ids: List[str] | None


class FinalCode(TypedDict):
    content: str
    sources: List[str] | None
    reflection: str | None


# ───────────────────── chunk-level structures ──────────────────── #

class RawChunk(TypedDict):
    """One 256-token chunk extracted from a GitHub file."""
    id:        str            # 8-char UUID
    url:       str            # original blob URL
    content:   str            # chunk text
    embedding: List[float]    # L2-normalised vector


class CodeChunk(TypedDict):
    """One 256-token chunk from the draft code."""
    id:        str
    url:       NotRequired[str | None]   # None for in-memory draft
    content:   str
    embedding: List[float]


# ─────────────────── similarity-ranking structure ───────────────── #

class ChunkMatch(TypedDict):
    """A raw chunk ranked as similar to a particular draft-code chunk."""
    url:     str
    content: str
    score:   float            # cosine similarity (0 → 1)


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
    initial_code:  NotRequired[InitialCode]
    final_content: NotRequired[FinalCode]

    # embeddings & similarity
    raw_chunks:   NotRequired[List[RawChunk]]
    code_chunks:  NotRequired[List[CodeChunk]]
    chunk_matches: NotRequired[Dict[str, List[ChunkMatch]]]   # key = code-chunk id

    # workflow meta
    status: str
