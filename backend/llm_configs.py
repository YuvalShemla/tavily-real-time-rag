# llm_configs.py
from dataclasses import dataclass
                     # holds Prompts.PLANNER, Prompts.SOLVER, ...

@dataclass(frozen=True)
class LLMParams:
    model: str
    temperature: float
    prompt: str                       # f-string with placeholders


#TODO improve prompts
class LLMConfig:

    PLANNER = LLMParams(
        model="gpt-4o",
        temperature=0.3,
        prompt="""
You are a world-class programming strategist.

TASK
1. Read the **user problem** from the user messege below and extract its core keywords.  
2. Produce a concise, ordered **solution_outline** (single string).  
3. Create **Three GitHub-focused search queries** that will help implement
   the outline.

STRICT RULES for each *search_queries* item  
• ≤ 12 words, no extra commentary or punctuation.   
• Queries must be distinct; each should target a different sub-task.

Return **only** the JSON block below: no Markdown, no fences, no commentary.

{
  "solution_outline": "<ordered plan, single string>",
  "search_queries": [
    "<query-1>",
    "<query-2>",
    "<query-3>"
  ]
}


""".strip()
)

    DRAFTER = LLMParams(
        model="gpt-4o",
        temperature=0.2,
        prompt="""

You are a world-class programmer.

**Your task**
Use the user messeges and the outline provided to write the complete, runnable solution in **one file**.

Rules
• The code must compile as-is (no “...”, no TODOs).  
• Use idiomatic style and minimal inline comments.  
• **Output the code ONLY — no markdown fences, no JSON, no commentary.**
""".strip(),
    )


    FILTER = LLMParams(
        model="gpt-4o",
        temperature=0.2,
        prompt="""
You are a code-search specialist.  
Your ONLY job is to pick 1-3 URLs from the list and return them in JSON.

Rules
1. **Pick from the provie url of the above docs.** 
2. **Return at least one URL from teh list only.** Choose the links that most directly help
   implement the outline.
3. choose urls which are code files, and provide as the slected url the url of the parant folder of the file.
   That it, remove the file name from the url.

3. Output JSON ONLY (no markdown, no comments no fences, no extra whitespace or chars).:

{
  "selected_urls": [
    "<url-1>",
    "<url-2>",
    "<url-3>"
  ]
}
""".strip(),
    )

    REFINER = LLMParams(
        model="gpt-4o",
        temperature=0.15,
        prompt=(
            "You are a world-class programmer. Your task is to refactor or rewrite the draft so "
            "it is correct, well-commented, use relevent information from the example code provided." 
            "comment the example URL above any block whose logic you borrow."
            "It should be a stand alone, runnable code, with no TODOs or incomplete parts."
        ),
    )

    FOLLOW_UP = LLMParams(
        prompt=(
            "You are an agent routing user requests.\n"
            "you previously provide a code solution to the user prvious problem"
            "Now the user asks for follow up question"
            "If the user expresses no additional need, reply with JSON:\n"
            '{"status": "done", "goodbye": "<goodluck messege spesific to the problem>"}\n'
            "Else rewrite the follow-up into a concise programming problem:\n"
            '{"status": "continue", "problem": "<rewritten>"}'
            "Return **only** the JSON block below: no Markdown, no fences, no commentary."
        ),
        model="gpt-4o",
        temperature=0.2,
    )
