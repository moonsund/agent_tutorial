import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()


class Source(BaseModel):
    name: str = Field(..., description="Source (for example Wikipedia or Web")
    detail: str = Field(..., description="link or article title or any other identification")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[Source]
    tools_used: list[str]


# -------------------------
# Tools
# -------------------------
search = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """Search the web for information (DuckDuckGo). Return a short, useful snippet."""
    q = (query or "").strip()
    if not q:
        return "Search error: empty query."
    try:
        return search.run(q)
    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"


wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information. Return a short extract."""
    q = (query or "").strip()
    if not q:
        return "Wikipedia error: empty query."
    try:
        return wiki.run(q)
    except Exception as e:
        return f"Wikipedia error: {type(e).__name__}: {e}"


tools = [search_web, wikipedia_search]


def save_text(data: str) -> str:
    """Save research result to file."""
    filename = "research_output.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n--- {timestamp} ---\n{data}\n")
    return f"Saved to {filename}"


# -------------------------
# LLM (через env, с дефолтами)
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# -------------------------
# Agent (structured output)
# -------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a research assistant.\n"
        "Rules:\n"
        "- Use tools when you need factual info.\n"
        "- Do not invent sources. Put only sources you actually used.\n"
        "- If you couldn't find sources, leave sources empty and say so in the summary.\n"
        "- Provide a concise, accurate summary.\n"
        "Output must match the given response schema."
    ),
    response_format=ResearchResponse,
)


# -------------------------
# Run
# -------------------------
messages = []

while True:
    user_query = input("What can I help you research? (type 'exit' to quit)").strip()
    if not user_query:
        raise SystemExit("Empty input. Please run again and enter a query.")
    if user_query.lower() in ['exit', 'quit', 'q']:
        break

    messages.append({"role": "user", "content": user_query})
    response = agent.invoke({"messages": messages})

    # structured output from the agent
    structured: ResearchResponse = response["structured_response"]

    # печать результата
    print(structured.model_dump_json(indent=2, ensure_ascii=False))
    messages.append({"role": "assistant", "content": structured.summary})

def _extract_tools_used(agent_response: dict) -> Optional[List[str]]:
    steps = agent_response.get("intermediate_steps")
    if not steps:
        return None
    used = []
    for step in steps:
        try:
            action = step[0]
            name = getattr(action, "tool", None) or getattr(action, "tool_name", None)
            if name:
                used.append(str(name))
        except Exception:
            pass
    uniq = []
    for x in used:
        if x not in uniq:
            uniq.append(x)
    return uniq or None

tools_used_real = _extract_tools_used(response)
if tools_used_real is not None:
    structured.tools_used = tools_used_real



# если нужно сохранить в файл — сохраняем явно (раскомментируйте)
# save_text(structured.model_dump_json(indent=2, ensure_ascii=False))