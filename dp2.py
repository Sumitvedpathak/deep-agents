import os
import datetime
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.tools import tool
from typing import Literal, Optional
from langchain_anthropic import ChatAnthropic


load_dotenv()
# llm_model = os.getenv("LLM_MODEL")

#to control thinking behavior, we can use the "thinking" parameter when creating the ChatAnthropic model.
model_name = os.getenv("LLM_MODEL").split(":", 1)[-1]
llm_model = ChatAnthropic(
    model=model_name, #sonnet5.5 doesnt support disabled thinking, so using haiku5.5 for now
    max_tokens=4000,
    thinking={"type": "disabled"}, #adaptive, disabled, or streaming
)

Category = Literal["leave", "expenses", "product", "onboarding"]

DOCS = {
    "hr-001": {
        "title": "Vacation and Time Off Policy",
        "category": "HR",
        "text": (
            "Employees accrue paid vacation based on tenure: 15 days per year "
            "for 0-2 years, 20 days for 3-5 years, 25 days after 5 years. "
            "To request time off, submit a request in the HR portal at least "
            "2 weeks in advance. Your manager approves or declines within 3 business days. "
            "Requests of more than 10 consecutive days need director approval."
        ),
    },
    "hr-002": {
        "title": "Sick Leave Policy",
        "category": "HR",
        "text": (
            "Employees get 10 paid sick days per year. No advance request is needed, "
            "but notify your manager before your shift starts. A doctor's note is "
            "required for absences longer than 3 days."
        ),
    },
    "exp-001": {
        "title": "Travel Expense Reimbursement",
        "category": "Expenses",
        "text": (
            "Submit expense reports within 30 days of travel. Receipts are required "
            "for any item over $25. Meals are reimbursed up to $60 per day. "
            "Flights must be booked through the company travel portal."
        ),
    },
    "onb-001": {
        "title": "New Hire Onboarding Checklist",
        "category": "Onboarding",
        "text": (
            "In week one: collect your laptop from IT, complete security training, "
            "set up your HR portal account, and meet your onboarding buddy."
        ),
    },
}

@tool
def search_docs(query:str, category:Optional[Category]=None)->str:
    """Search the Northwind internal knowledge base (HR, expenses, product, onboarding).
Args:
    query: 2-6 keywords, not a full sentence (e.g. "vacation days tenure").
    category: Restrict to one category. Omit to search all.
Returns:
    Up to 5 matches, each as: doc_id | title | one-line snippet.
    Use get_doc(doc_id) to read the full text. Returns "No matches" if empty.
    """
    print(f"tool called--query: {query}, category: {category}")
    words = query.lower().split()
    scored = []
    for doc_id, d in DOCS.items():
        if category and d["category"].lower() != category.lower():
            continue
        haystack = (d["title"] + " " + d["text"]).lower()
        score = sum(w in haystack for w in words)
        if score:
            scored.append((score, doc_id, d))
    if not scored:
        return "No matches"
    scored.sort(key=lambda x: -x[0])
    return "\n".join(
        f"{doc_id} | {d['title']} | {d['text'][:80]}..."
        for _, doc_id, d in scored[:5]
    )

@tool
def get_doc(doc_id:str)->str:
    """Get a specific document by ID.
    Args:
        doc_id: The ID of the document from which to retrieve.
    Returns:
        The requested document.
    """
    print(f"tool called--doc_id: {doc_id}")

    d = DOCS.get(doc_id)
    return d["text"] if d else f"No document with id {doc_id}"


system_prompt = f"""You are a research assistant for Northwind, a fictional company. Today's date is {datetime.date.today()}.
You have access to the Northwind internal knowledge base, which contains information about HR policies, expenses, product details, 
and onboarding procedures. You can use the following tools to assist you in finding information:
You can search the relevant documents from search_docs(query, category) and retrieve the full text of a document using get_doc(doc_id).
When answering questions, please provide accurate and concise information based on the available documents. 
If you dont known the answer, say so rather than guessing. Answer in short, conversational prose with no markdown, bullets or bold. Name the document you used."""

agent = create_deep_agent(
    model=llm_model,
    tools=[search_docs, get_doc],
    system_prompt=system_prompt
)

question = "How to apply for a claim?"
inputs = {"messages": [{"role": "user", "content": question}]}
response = agent.invoke(inputs)
print(response["messages"][-1].content)


# if isinstance(response["messages"][-1].content, str):
#     print(response["messages"][-1].content)
# else:
#     print("".join(b["text"] for b in response["messages"][-1].content if b.get("type") == "text"))