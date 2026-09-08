"""
Fact-checker / critic agent.

This is the node that makes the system feel trustworthy rather than a
thin LLM wrapper. It looks across ALL findings gathered by the parallel
research agents and flags anything that's:
  - claimed by only one source, or
  - contradicted by another source.

For your eval harness: seed a test run with a deliberately wrong or
single-sourced claim and confirm this node catches it. That's the
concrete, measurable claim you can put in an interview.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import GraphState
from agents.utils import call_llm, get_text

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

SYSTEM_PROMPT = """You are a fact-checking agent. You will be given a list of \
claims gathered by research agents, each with a source. Identify any claims \
that are:
1. Contradicted by another claim in the list, or
2. Supported by only one source and stated as if it were settled fact.

Respond with ONLY a list of the flagged claims (verbatim), one per line. \
If none are flagged, respond with "NONE"."""


def fact_checker_node(state: GraphState) -> dict:
    findings_text = "\n".join(
        f"- {f['claim']} (source: {f['citation']})" for f in state["findings"]
    )

    response = call_llm(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": findings_text},
    ], node_name="fact_checker")

    content = get_text(response)
    flagged = [] if content == "NONE" else [
        line.strip("- ").strip() for line in content.split("\n") if line.strip()
    ]

    return {"flagged_findings": flagged}