"""
Orchestrator agent.

Its only job: take the raw user question and split it into 2-4
sub-questions that the parallel research agents can tackle independently.
Keeping this agent narrow (one responsibility) makes the graph easier to
debug and eval -- if routing is wrong, you know to look here first.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import GraphState
from agents.utils import call_llm, get_text

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

SYSTEM_PROMPT = """You are a research orchestrator. Given a business research \
question, split it into 2-4 focused sub-questions that, together, fully \
answer the original question. Each sub-question should be answerable by a \
web search or document search.

Respond with ONLY a numbered list, one sub-question per line. No preamble."""


def orchestrator_node(state: GraphState) -> dict:
    response = call_llm(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["question"]},
    ], node_name="orchestrator")

    lines = [
        line.split(".", 1)[-1].strip()
        for line in get_text(response).split("\n")
        if line.strip()
    ]

    return {"sub_questions": lines}