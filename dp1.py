import os
import datetime
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.tools import tool
from tavily import TavilyClient
# from langchain_core.callbacks import BaseCallbackHandler


load_dotenv()  # Load environment variables from .env file

tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information.

    Args:
        query: A specific search query (3-8 words works best). Use one
            focused question per call rather than a long compound query.
        max_results: Number of results to return (1-10).

    Returns:
        Results as text, each with title, URL and a content snippet.
        Returns an error message string if the search fails.
    """
       # Format the results into a string
    print(f"tool called--query: {query}")

    try:
        resp = tavily.search(
            query=query,
            max_results=max(1, min(max_results, 10)),
            search_depth="basic",
        )
    except Exception as e:
        return f"Search failed: {e}. Try rephrasing the query or a different angle."

    results = resp.get("results", [])
    if not results:
        return "No results found. Try a different query."

    return "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}"
        for i, r in enumerate(results, 1))

# class ShowPrompt(BaseCallbackHandler):
#     def on_chat_model_start(self, serialized, messages, **kwargs):
#         for m in messages[0]:
#             if m.type == "system":
#                 blocks = m.content if isinstance(m.content, list) else [m.content]
#                 print(f"system message: {len(blocks)} block(s)")
#                 for i, b in enumerate(blocks):
#                     text = b if isinstance(b, str) else b.get("text", "")
#                     print(f"--- block {i}: {len(text)} chars ---\n{text[:300]}...")
#                 break


SYSTEM_PROMPT = f"""You are a web research assistant. Today's date is {datetime.date.today()}.
- Break the question into focused searches; run several, refining queries based on what you find.
- Stop when two independent sources agree, or after about 6 searches.
- Prefer official docs and recent sources.
- Cite the URL for every claim. If you can't find something, say so plainly rather than guessing.
- Finish with a short answer first, then supporting details."""
print("Hello from deep-agents!")
agent = create_deep_agent(model=os.getenv("LLM_MODEL"), 
                        tools=[web_search], 
                        system_prompt=SYSTEM_PROMPT)

question = "What is the current state of AI research in 2026? Please provide a summary of recent advancements and notable papers."
inputs = {"messages": [{"role": "user", "content": question}]}
result = agent.invoke(inputs)
print(result["messages"][-1].content)

# agent.invoke(inputs, config={"callbacks": [ShowPrompt()]})


for chunk in agent.stream(inputs, stream_mode="updates"):
        for node, update in chunk.items():
            for msg in (update or {}).get("messages", []):
                for call in getattr(msg, "tool_calls", None) or []:
                    print(f"[{node}] tool call: {call['name']}({call['args']})")

