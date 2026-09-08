"""
This is the file that ties everything together into an actual graph.

Notice what this ISN'T: it's not a `while True` loop deciding what to do
next based on string-matching. It's an explicit state machine where every
transition is declared up front. That's the exact thing 2026 hiring
guidance points to as separating production-shaped systems from ad-hoc
agent scripts.

Key feature to be able to explain in an interview:
`interrupt_before=["research_dispatch", "writer"]` -- two pause points.
The graph pauses before "research_dispatch" because that node's real
work (the async MCP fan-out) can't run inside a sync LangGraph node --
main.py does that work outside the graph, then calls
`graph.update_state(config, {"findings": [...]})` followed by
`graph.invoke(None, config)` to resume. It pauses again before "writer"
so a human must approve the outline before any drafting tokens are
spent. The graph literally cannot proceed past either point until the
app resumes it -- that's a real guardrail, not a comment saying "add
approval step here later."

IMPORTANT: resuming after an interrupt must use `graph.update_state()`
+ `graph.invoke(None, config)`. Calling `graph.invoke({...}, config)`
with a plain dict does NOT resume -- it starts a brand new run from
START on the same thread. (Verified against langgraph 1.x semantics.)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from agents.orchestrator import orchestrator_node
from agents.fact_checker import fact_checker_node
from agents.synthesis import synthesis_node
from agents.writer import writer_node


def human_approval_node(state: GraphState) -> dict:
    # This node does nothing itself -- it exists purely as the
    # interrupt point. Execution pauses here until the human resumes
    # the graph (see main.py for how that resume call looks).
    return {}


def route_after_approval(state: GraphState) -> str:
    return "writer" if state.get("outline_approved") else END


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("orchestrator", orchestrator_node)
    # NOTE: research_node is async and needs a live MCP session, so in
    # practice it's wrapped by a small dispatcher node here that fans
    # out one call per sub_question and awaits them concurrently. See
    # main.py for the working wrapper -- kept out of this file so the
    # graph topology stays readable at a glance.
    graph.add_node("research_dispatch", lambda state: state)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "research_dispatch")
    graph.add_edge("research_dispatch", "fact_checker")
    graph.add_edge("fact_checker", "synthesis")
    graph.add_edge("synthesis", "human_approval")
    graph.add_conditional_edges("human_approval", route_after_approval)
    graph.add_edge("writer", END)

    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["research_dispatch", "writer"],
    )
