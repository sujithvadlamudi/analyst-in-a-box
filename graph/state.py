"""
The shared state object that flows through every node in the graph.

In LangGraph, every node reads from this state and returns a partial
update to it -- the graph merges updates back in. Keeping state explicit
like this (rather than passing loose variables between functions) is
exactly the "state machine, not a while-loop" pattern that separates a
production-shaped agent system from an ad-hoc script.
"""

from typing import TypedDict, Annotated
import operator


class ResearchFinding(TypedDict):
    source: str          # which research agent produced this
    claim: str            # the specific claim/fact found
    citation: str          # url or reference


class GraphState(TypedDict):
    # the original user question
    question: str

    # sub-questions the orchestrator splits the question into
    sub_questions: list[str]

    # raw findings from each parallel research agent
    # Annotated with operator.add so parallel branches can each append
    # to this list without overwriting each other's results
    findings: Annotated[list[ResearchFinding], operator.add]

    # findings the fact-checker flagged as single-sourced or contradicted
    flagged_findings: list[str]

    # the structured outline produced by the synthesis agent
    outline: str

    # set to True once the human approves the outline
    outline_approved: bool

    # the final written report
    final_report: str
