**Tavily Dynamic RAG**

**Motivation**  
LLMs are great writers when the knowledge they need is already in their weights or widely available on the web. They stumble when the answer is **new, niche, or buried**.  
Retrieval-Augmented Generation (RAG) fixes that by injecting fresh documents into the prompt… but classic RAG assumes you already have a data-store containing every doc a user might need.

**Dynamic RAG** would help gather fresh the sources for each query.
With Tavily’s search + crawl APIs we can pull dozens or hundreds of raw files in about a minute, feed them to an LLM, and return:

* a richer, more accurate answer  
* clickable source links so the user can dig deeper  

This repo is a template, and the goal was to make it adaptable to any domain. The included example is a **coding RAG**: given a problem, it finds relevant GitHub files, learns from them, and refines its solution.

---

**How it works**  
the workflow is implemented using LangGraph; each node owns a single job:

| Phase      | Node      | What it does                                                             |
|------------|-----------|--------------------------------------------------------------------------|
| Planning   | Planner   | LLM drafts an outline and up-to-3 Tavily search queries                  |
| Drafting   | Drafter   | First code attempt                                                      |
| Search     | Search    | Tavily Search → candidate URLs                                          |
| Crawl      | Crawler   | Tavily Crawl → raw pages/files                                          |
| Extract    | Extract   | Converts GitHub `/blob/` → `raw` and grabs file text                    |
| Ranking    | Ranker    | Embeds draft + files, computes cosine similarity                        |
| Refining   | Refiner   | Rewrites draft using the top-K files as examples                        |
| Responding | Responder | Shows result, asks user if they need more                               |

*(Diagram here)*

**Key points**

* **Search → Crawl** gives breadth *and* depth—great for APIs or new packages the model hasn’t seen.  
* **Similarity ranking** uses `text-embedding-3-small` on the first 8 000 chars of each file.  
* **Extract** is domain-specific (GitHub). Swap it out for another spesific downloader or skip it entirely since in most cases the crawl already returns the relavent raw text.

---

**Tweaking & improving**

* **Prompts** – All prompt text lives in `llm_configs.py`. Tuning them (or switching models) has the biggest impact. Improtant to keep the JSON output formats intact.  
* **Embeddings & Chunking** – Right now the embedding is one 8 k-char slice per file. For lbetter results: chunk + overlap + rank chunks. A local model like CodeBERT or other spesific embeders would be ideal, but on a local run on my (old) MacBook it was too slow.  
* **Crawl parameters** – Current params are (`max_depth=3`, `max_breadth=100`, `limit=500`) are a compromise between speed and coverage for GitHub. Different domains will want different params.


