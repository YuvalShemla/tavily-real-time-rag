## Tavily Dynamic RAG

**Motivation**  
LLMs are great writers when the knowledge they need is already in their weights or widely available on the web. They stumble when the answer is **new, niche, or buried**.  
Retrieval-Augmented Generation (RAG) fixes that by injecting fresh documents into the prompt… but classic RAG assumes you already have a data-store containing every doc a user might need.

**Dynamic RAG** would help gather fresh the sources for each query.
With Tavily’s search + crawl APIs we can pull dozens or hundreds of raw files in about a minute, feed them to an LLM, and return:

* a richer, more accurate answer  
* clickable source links so the user can dig deeper  

This repo is a template, and the goal was to make it adaptable to any domain. The included example is a **coding RAG**: given a problem, it finds relevant GitHub files, learns from them, and refines its solution.

---

**Project Overview**  
the workflow is implemented using LangGraph; each node owns a single job:

| Node      | What it does                                                             |
|-----------|--------------------------------------------------------------------------|
| Planner   | LLM drafts an outline and up-to-3 Tavily search queries                  |
| Drafter   | First code attempt                                                       |
| Search    | Tavily Search → candidate URLs                                           |
| Crawler   | Tavily Crawl → raw pages/files                                           |
| Extract   | Converts GitHub `/blob/` → `raw` and grabs file text                     |
| Ranker    | Embeds draft + files, computes cosine similarity                         |
| Refiner   | Rewrites draft using the top-K files as examples                         |
| Responder | Displays the result, collects user feedback, and decides whether to run another cycle or finish |

*(Diagram here)*

**Key points**

* **Search → Crawl** gives breadth *and* depth—great for APIs or new packages the model hasn’t seen.  
* **Similarity ranking** uses `text-embedding-3-small` on the first 8 000 chars of each file.  
* **Extract** is domain-specific (GitHub). Swap it out for another spesific downloader or skip it entirely since in most cases the crawl already returns the relavent raw text.

---

## Repository structure

tavily_dynamic_rag/
├── backend/
│   ├── __init__.py
│   ├── base_node.py
│   ├── llm_configs.py
│   ├── main.py
│   ├── state.py
│   ├── utils.py
│   ├── logs/
│   │   └── backend.log
│   └── nodes/
│       ├── __init__.py
│       ├── planner.py
│       ├── drafter.py
│       ├── search.py
│       ├── crawl.py
│       ├── extract.py
│       ├── ranker.py
│       ├── refiner.py
│       ├── filter.py
│       └── responder.py
├── results/
│       ├── example_1.py
│       ├── example_2.py
│       ├── example_3.py
│       ├── example_rec.mp4
├── requirements.txt
└── README.md

---

## Quick start

1. **Clone & set up a virtual-env**

   git clone https://github.com/your-name/tavily-in-the-loop.git
   cd tavily_dynamic_rag
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate

2. **Install dependencies**
   pip3 install -r requirements.txt

3. **Add API keys**

    create .env file and set:
    TAVILY_API_KEY=...
    OPENAI_API_KEY=...

## Next Steps & Inprovments 

* **Prompts** – All prompt text lives in `llm_configs.py`. Tuning them (or switching models) has the biggest impact. Improtant to keep the JSON output formats intact.  
* **Embeddings & Chunking** – Right now the embedding is one 8 k-char slice per file. For lbetter results: chunk + overlap + rank chunks. A local model like CodeBERT or other spesific embeders would be ideal, but on a local run on my (old) MacBook it was too slow.  
* **Crawl parameters** – Current params are (`max_depth=3`, `max_breadth=100`, `limit=500`) are a compromise between speed and coverage for GitHub. Different domains will want different params.
* **Multi-loop interaction** – The workflow already supports iterative refinement: after each answer, the user can ask follow-up questions and trigger another cycle. Currently, we only carry over the latest user prompt and the responder’s message. A next step would be to persist a compact history (e.g., outline snapshots, top sources, or key code diffs) so each loop has richer context without bloating the prompt.

