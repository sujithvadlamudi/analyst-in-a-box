"""
Web UI for Analyst-in-a-Box, built on Streamlit.

This wraps the exact same LangGraph pipeline used by main.py -- same
graph, same interrupt/resume pattern, same retry logic in agents/utils.py
-- just swapping terminal input()/print() for buttons and text areas so
it can be deployed as a live demo (e.g. Streamlit Community Cloud)
instead of something a recruiter has to clone and run themselves.

Run locally with:
    streamlit run streamlit_app.py
(reads GOOGLE_API_KEY and TAVILY_API_KEY from .env, same as main.py)

Deployed, it reads both from Streamlit's Secrets manager instead --
see the README for setup.
"""

import os
import uuid

import streamlit as st

# Secrets must be copied into the environment BEFORE importing anything
# that constructs a ChatGoogleGenerativeAI client (agents/*.py do this
# at import time), since the client reads GOOGLE_API_KEY from the
# environment at construction. Wrapped in try/except because accessing
# st.secrets raises if no secrets.toml exists yet (e.g. local dev,
# where .env + load_dotenv() is used instead -- see main.py).
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ.setdefault("GOOGLE_API_KEY", st.secrets["GOOGLE_API_KEY"])
    if "TAVILY_API_KEY" in st.secrets:
        os.environ.setdefault("TAVILY_API_KEY", st.secrets["TAVILY_API_KEY"])
except Exception:
    pass

import asyncio

from main import run_research_fanout
from graph.build_graph import build_graph
from agents.utils import AgentError

st.set_page_config(page_title="Analyst-in-a-Box", page_icon="📊")


@st.cache_resource
def get_graph():
    return build_graph()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "idle"  # idle -> researching -> outline -> done | rejected | error

graph = get_graph()
config = {"configurable": {"thread_id": st.session_state.session_id}}

st.title("Analyst-in-a-Box")
st.caption(
    "Multi-agent research pipeline: orchestrator -> parallel research agents "
    "-> fact-checker -> synthesis -> your approval -> writer."
)

if not os.environ.get("GOOGLE_API_KEY"):
    st.error(
        "GOOGLE_API_KEY is not set. If you're running this locally, add it "
        "to a .env file. If this is the deployed app, add it under "
        "Settings -> Secrets."
    )
    st.stop()

if not os.environ.get("TAVILY_API_KEY"):
    st.error(
        "TAVILY_API_KEY is not set. If you're running this locally, add it "
        "to a .env file. If this is the deployed app, add it under "
        "Settings -> Secrets. Get a free key (no credit card) at "
        "https://tavily.com."
    )
    st.stop()

question = st.text_input(
    "Business research question",
    placeholder="Compare the top 3 CRM vendors for a 50-person startup",
    disabled=st.session_state.stage not in ("idle", "rejected", "error"),
)

if st.session_state.stage in ("idle", "rejected", "error"):
    if st.button("Research", type="primary", disabled=not question):
        st.session_state.stage = "researching"
        st.session_state.error_message = None
        with st.spinner("Splitting the question and researching in parallel..."):
            try:
                state = graph.invoke({"question": question, "findings": []}, config)
                findings = asyncio.run(run_research_fanout(state["sub_questions"]))
                if not findings:
                    st.session_state.stage = "error"
                    st.session_state.error_message = (
                        "Research failed twice in a row for every sub-question. "
                        "Check the network and GOOGLE_API_KEY, then try again."
                    )
                else:
                    graph.update_state(config, {"findings": findings})
                    state = graph.invoke(None, config)
                    st.session_state.outline = state["outline"]
                    st.session_state.stage = "outline"
            except AgentError as exc:
                st.session_state.stage = "error"
                st.session_state.error_message = str(exc)
        st.rerun()

if st.session_state.stage == "error":
    st.error(st.session_state.error_message)

if st.session_state.stage == "outline":
    st.subheader("Proposed outline")
    st.markdown(st.session_state.outline)
    col1, col2 = st.columns(2)
    if col1.button("Approve and write report", type="primary"):
        with st.spinner("Writing the final report..."):
            try:
                graph.update_state(config, {"outline_approved": True})
                final_state = graph.invoke(None, config)
                st.session_state.final_report = final_state["final_report"]
                st.session_state.stage = "done"
            except AgentError as exc:
                st.session_state.stage = "error"
                st.session_state.error_message = str(exc)
        st.rerun()
    if col2.button("Reject and start over"):
        st.session_state.stage = "rejected"
        st.rerun()

if st.session_state.stage == "done":
    st.subheader("Final report")
    st.markdown(st.session_state.final_report)
    st.download_button(
        "Download as markdown",
        st.session_state.final_report,
        file_name="report.md",
    )
    if st.button("Start a new research question"):
        st.session_state.stage = "idle"
        st.rerun()