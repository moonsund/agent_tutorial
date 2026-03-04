"""
Improved LangChain tool-calling research agent

What’s improved vs your original:
- Uses StructuredTool (typed tool schema via Pydantic)
- Uses OutputFixingParser (auto-fixes malformed JSON output)
- Adds retry logic around execution + parsing
- Automatically saves the *structured* response
- return_intermediate_steps=True so you can inspect tool calls

Requirements (typical):
pip install langchain langchain-openai langchain-community pydantic python-dotenv duckduckgo-search wikipedia

.env should contain:
OPENAI_API_KEY=...
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# OutputFixingParser location differs across LangChain versions; handle both.
try:
    from langchain.output_parsers import OutputFixingParser  # older path
except Exception:
    from langchain_core.output_parsers import OutputFixingParser  # newer path (some builds)

from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import StructuredTool


# ----------------------------
# 1) Load env
# ----------------------------
load_dotenv()


# ----------------------------
# 2) Pydantic schemas
# ----------------------------
class ResearchResponse(BaseModel):
    topic: str = Field(..., description="Main topic of the research")
    summary: str = Field(..., description="Concise research summary")
    sources: List[str] = Field(default_factory=list, description="List of sources (URLs or citations)")
    tools_used: List[str] = Field(default_factory=list, description="Which tools were used")


class SaveInput(BaseModel):
    data: str = Field(..., description="Text content to save")
    filename: str = Field(default="research_output.txt", description="Target filename to append to")


# ----------------------------
# 3) Tools (StructuredTool)
# ----------------------------
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"


save_tool = StructuredTool.from_function(
    name="save_text_to_file",
    description="Saves text to a file (appends). Useful for persisting the final structured result.",
    func=save_to_txt,
    args_schema=SaveInput,
)

search = DuckDuckGoSearchRun()
search_tool = StructuredTool.from_function(
    name="search",
    description="Search the web for information using DuckDuckGo.",
    func=search.run,
)

api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)  # already a Tool-like runnable


# ----------------------------
# 4) LLM + Parser (+ Fixing)
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base_parser = PydanticOutputParser(pydantic_object=ResearchResponse)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research assistant.
- Use tools when needed (search, wikipedia).
- Return ONLY valid JSON matching the provided schema. No extra text.
{format_instructions}
            """.strip(),
        ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=base_parser.get_format_instructions())


# ----------------------------
# 5) Agent + Executor (intermediate steps enabled)
# ----------------------------
tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,  # ✅ lets you see tool calls + observations
)


# ----------------------------
# 6) Helper: extract output text robustly across versions
# ----------------------------
def extract_output_text(raw_response: dict) -> str:
    """
    Different LangChain versions return output differently.
    Common patterns:
      - raw_response["output"] is a string
      - raw_response["output"] is a list of message blocks with ["text"]
    """
    if raw_response is None:
        raise ValueError("No response from agent executor")

    out = raw_response.get("output")
    if isinstance(out, str):
        return out

    if isinstance(out, list) and out:
        # Often something like: [{"text": "..."}]
        first = out[0]
        if isinstance(first, dict) and "text" in first and isinstance(first["text"], str):
            return first["text"]

    # fallback: try to stringify
    return json.dumps(out, ensure_ascii=False)


# ----------------------------
# 7) Run with retry + auto-save structured result
# ----------------------------
def run_research(query: str, max_attempts: int = 3) -> dict:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = agent_executor.invoke({"query": query})

            output_text = extract_output_text(raw)

            # First try strict parse; if fails, auto-fix via LLM
            try:
                structured: ResearchResponse = base_parser.parse(output_text)
            except Exception:
                structured = fixing_parser.parse(output_text)

            # Auto-save the structured JSON (pretty printed)
            saved_payload = structured.model_dump()
            save_to_txt(
                data=json.dumps(saved_payload, ensure_ascii=False, indent=2),
                filename="research_output.txt",
            )

            return {
                "structured": saved_payload,
                "intermediate_steps": raw.get("intermediate_steps", []),
                "raw_output_text": output_text,
            }

        except Exception as e:
            last_err = e
            # retry
            continue

    raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_err}")


if __name__ == "__main__":
    user_query = input("What can I help you research? ").strip()
    result = run_research(user_query, max_attempts=3)

    print("\n=== STRUCTURED RESULT ===")
    print(json.dumps(result["structured"], ensure_ascii=False, indent=2))

    print("\n=== INTERMEDIATE STEPS (tool calls) ===")
    # Keep it readable (intermediate steps can be large)
    for i, step in enumerate(result["intermediate_steps"], start=1):
        print(f"\n--- Step {i} ---")
        print(step)