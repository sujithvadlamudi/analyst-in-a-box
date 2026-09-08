"""
Writer agent.

Only runs after the human has approved the outline (see graph/build_graph.py
for the interrupt that enforces this). Turns the approved outline plus the
verified findings into a full, cited report.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import GraphState
from agents.utils import call_llm, get_text

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.3)

SYSTEM_PROMPT = """You are a business report writer. Given an approved outline \
and a list of verified findings with citations, write a polished report. \
Cite sources inline as (Source: <url>). Keep it factual and avoid speculation \
beyond what the findings support."""


def writer_node(state: GraphState) -> dict:
    findings_text = "\n".join(
        f"- {f['claim']} (Source: {f['citation']})" for f in state["findings"]
    )

    response = call_llm(llm, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Approved outline:\n{state['outline']}\n\n"
                       f"Findings:\n{findings_text}",
        },
    ], node_name="writer")

    return {"final_report": get_text(response)}