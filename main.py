"""
Entry point. Run with:
    python main.py "Compare the top 3 CRM vendors for a 50-person startup"

Walks through:
  1. Start the graph -> orchestrator splits into sub-questions
  2. Fan out one research_node call per sub-question, in parallel,
     each hitting the web_search MCP server
  3. Fact-checker + synthesis run automatically
  4. Graph PAUSES at the human_approval interrupt -- outline is
     printed, you approve or reject in the terminal
  5. If approved, graph resumes and the writer agent drafts the report
"""

import asyncio
import sys
from pathlib import Path

# Force line-buffered stdout. Without this, print() output can sit in a
# buffer and never reach hosted-platform log viewers (e.g. Streamlit
# Cloud) until the process exits -- which made earlier debugging on that
# platform show generic library log lines but none of this project's own
# diagnostic prints (the [research: ...] retry/failure messages below).
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from .env into the environment

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from graph.build_graph import build_graph
from agents.research import research_node
from agents.utils import AgentError

# Absolute path to the MCP server script, resolved relative to this file --
# not the current working directory. A relative path like
# "mcp_servers/web_search_server.py" only works if the process is launched
# from the project root; hosted platforms (e.g. Streamlit Cloud) don't
# guarantee that, so this fails silently there otherwise.
_MCP_SERVER_PATH = Path(__file__).parent / "mcp_servers" / "web_search_server.py"


async def _staggered_research(sub_question: str, mcp_session, delay: float) -> dict:
    # Small stagger between the parallel research agents' *first* LLM
    # call. Firing 3-4 calls in the exact same instant is itself enough
    # to spike past the free tier's per-minute limit, even before any
    # retries happen. This keeps the burst rate down without giving up
    # real concurrency -- the searches and later calls still overlap.
    if delay:
        await asyncio.sleep(delay)
    return await research_node({}, sub_question, mcp_session)


async def _fanout_once(sub_questions: list[str]) -> list[dict]:
    server_params = StdioServerParameters(
        # sys.executable, not the string "python" -- guarantees the exact
        # same interpreter (and venv) that's running this process is used
        # to launch the MCP server subprocess. On some hosted platforms
        # (e.g. Streamlit Cloud's containers) there's no "python" on PATH,
        # only "python3" or a specific versioned binary, so a bare
        # "python" command fails with FileNotFoundError.
        command=sys.executable, args=[str(_MCP_SERVER_PATH)]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tasks = [
                _staggered_research(q, session, delay=i * 3)
                for i, q in enumerate(sub_questions)
            ]
            # return_exceptions=True: research_node already catches its
            # own errors and returns {"findings": []} on failure, but
            # this is a second safety net -- if something still slips
            # through (e.g. the MCP server process itself dies), one
            # bad task must not take down the other sub-questions'
            # results via asyncio.gather's default fail-fast behavior.
            results = await asyncio.gather(*tasks, return_exceptions=True)

    findings = []
    for q, r in zip(sub_questions, results):
        if isinstance(r, Exception):
            print(f"  [research: {q!r}] unhandled failure: {r}")
            continue
        findings.extend(r["findings"])
    return findings


async def run_research_fanout(sub_questions: list[str]) -> list[dict]:
    """Redirect-back-on-failure: if every sub-question came back with
    zero findings (e.g. the network was down, or the MCP server failed
    to start), that's not "research succeeded with nothing to say" --
    it's a failed batch. Retry the whole fan-out once before giving up,
    rather than silently letting fact-checker/synthesis run on nothing.
    """
    findings = await _fanout_once(sub_questions)
    if not findings:
        print("  [research] all sub-questions came back empty -- retrying batch once...")
        findings = await _fanout_once(sub_questions)
    return findings


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Compare the top 3 CRM vendors for a 50-person startup"
    )
    graph = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}

    try:
        # Run 1: START -> orchestrator -> pauses before research_dispatch
        # (findings must start as [] since it's an operator.add field)
        state = graph.invoke({"question": question, "findings": []}, config)

        # research_dispatch is a no-op node -- it exists purely as the
        # interrupt point. The real parallel fan-out happens here,
        # outside the graph, because it needs an async MCP session.
        findings = asyncio.run(run_research_fanout(state["sub_questions"]))

        if not findings:
            print(
                "\nResearch failed twice in a row for every sub-question "
                "(check your network and GOOGLE_API_KEY). Stopping before "
                "fact-checking, since there's nothing to check."
            )
            return

        # Resume: inject findings into checkpointed state, then continue
        # from research_dispatch -> fact_checker -> synthesis ->
        # human_approval -> (routes to END since outline_approved is False)
        graph.update_state(config, {"findings": findings})
        state = graph.invoke(None, config)

        print("\n=== Proposed outline ===\n")
        print(state["outline"])

        approval = input("\nApprove this outline and draft the full report? [y/N] ")
        if approval.strip().lower() == "y":
            # Resume again: this time the run reaches the writer
            # interrupt and continues through it since outline_approved
            # is now True.
            graph.update_state(config, {"outline_approved": True})
            final_state = graph.invoke(None, config)
            print("\n=== Final report ===\n")
            print(final_state["final_report"])
        else:
            print("Outline rejected -- stopping before draft.")

    except AgentError as exc:
        # Raised by agents/utils.call_llm once an agent node has
        # exhausted its retries (orchestrator, fact-checker, synthesis,
        # or writer). Printed cleanly instead of a raw traceback --
        # the graph's checkpointed state is preserved on thread
        # "session-1" if you want to inspect it before rerunning.
        print(f"\nStopped: {exc}")
        print("This was a Gemini API or network failure, not a bug in the "
              "graph logic. Try again in a moment, or check your "
              "GOOGLE_API_KEY / rate limits.")


if __name__ == "__main__":
    main()