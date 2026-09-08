"""
Research agent.

Runs once per sub-question, in parallel (LangGraph fans these out
automatically because the state's `findings` field uses operator.add --
see graph/state.py). Each instance calls the web_search MCP tool and
asks the LLM to extract discrete, citable claims from the results.

This is the piece that makes the "self-authored MCP server" claim
concrete: this agent doesn't call DuckDuckGo directly, it calls the
MCP tool defined in mcp_servers/web_search_server.py.

ERROR HANDLING: this function is fanned out N times concurrently (one
per sub-question) by main.py's asyncio.gather. If sub-question 2 of 4
fails, that must NOT take down sub-questions 1, 3, and 4 -- so every
failure here is caught and turned into an empty findings list rather
than an exception. main.py's fanout also catches exceptions defensively
as a second safety net, and retries the *whole batch once* if every
single sub-question came back empty (see run_research_fanout).
"""

import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import GraphState
from agents.utils import call_llm, get_text

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

EXTRACTION_PROMPT = """Given these search results for the question "{question}", \
extract 2-3 specific, factual claims. For each claim, note the source URL.

Search results:
{results}

Respond as a list in the format:
CLAIM: <the claim>
SOURCE: <url>
"""


async def research_node(
    state: GraphState, sub_question: str, mcp_session, max_retries: int = 2
) -> dict:
    # mcp_session is an active MCP ClientSession connected to
    # web_search_server.py -- call the tool exactly like any other
    # MCP-exposed capability.
    results = None
    last_error = None

    for attempt in range(1, max_retries + 2):
        try:
            tool_result = await mcp_session.call_tool(
                "web_search", arguments={"query": sub_question, "max_results": 5}
            )
            # call_tool returns an mcp.types.CallToolResult, not a plain
            # list -- iterating it directly yields (field_name, value)
            # pydantic tuples, not result dicts. The actual list lives
            # in .structuredContent, which FastMCP populates
            # automatically from the `-> list[dict]` return annotation.
            if tool_result.isError:
                raise RuntimeError(f"web_search tool error: {tool_result.content}")
            results = tool_result.structuredContent["result"]
            break
        except Exception as exc:
            last_error = exc
            if attempt <= max_retries:
                wait = 2 ** (attempt - 1)
                print(
                    f"  [research: {sub_question!r}] search failed "
                    f"({exc.__class__.__name__}), retrying in {wait}s "
                    f"(attempt {attempt}/{max_retries})..."
                )
                await asyncio.sleep(wait)

    if results is None:
        # Every attempt failed. Don't raise -- that would take down the
        # whole asyncio.gather() and every OTHER sub-question's results
        # along with it. Return no findings for this one sub-question
        # and let the rest of the fan-out succeed independently.
        print(
            f"  [research: {sub_question!r}] giving up after "
            f"{max_retries + 1} attempts: {last_error}"
        )
        return {"findings": []}

    results_text = "\n".join(
        f"- {r['title']}: {r['body']} ({r['href']})" for r in results
    )

    try:
        response = call_llm(
            llm,
            [{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    question=sub_question, results=results_text
                ),
            }],
            node_name=f"research: {sub_question!r}",
            max_retries=max_retries,
        )
    except Exception as exc:
        # Same reasoning as above: a claim-extraction failure for one
        # sub-question shouldn't sink the others.
        print(f"  [research: {sub_question!r}] extraction failed: {exc}")
        return {"findings": []}

    findings = []
    for block in get_text(response).split("CLAIM:")[1:]:
        claim_part, _, source_part = block.partition("SOURCE:")
        findings.append({
            "source": "web_research",
            "claim": claim_part.strip(),
            "citation": source_part.strip().split("\n")[0],
        })

    return {"findings": findings}