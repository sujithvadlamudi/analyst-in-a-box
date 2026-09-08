"""
Synthesis agent.

Turns verified findings into a structured outline. This runs BEFORE the
human approval checkpoint and BEFORE the writer agent -- the whole point
is to get a cheap, short artifact (an outline) in front of a human before
spending tokens on a full draft. This is the "production cost control"
detail worth calling out in an interview.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import GraphState
from agents.utils import call_llm, get_text

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

SYSTEM_PROMPT = """You are a research synthesis agent. Given verified findings \
and a list of any flagged/uncertain claims, produce a short structured outline \
(section headers + one-line description each) for a business research report. \
Do not write full prose yet -- this is just the outline for human review.

Note any flagged claims explicitly as "needs verification" in the relevant section."""


def synthesis_node(state: GraphState) -> dict:
    verified = [
        f for f in state["findings"] if f["claim"] not in state.get("flagged_findings", [])
    ]
    findings_text = "\n".join(f"- {f['claim']}" for f in verified)
    flagged_text = "\n".join(state.get("flagged_findings", [])) or "None"

    response = call_llm(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Verified findings:\n{findings_text}\n\n"
                       f"Flagged (needs verification):\n{flagged_text}",
        },
    ], node_name="synthesis")

    return {"outline": get_text(response), "outline_approved": False}