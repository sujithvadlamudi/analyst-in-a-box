# Analyst-in-a-Box

A multi-agent research system: give it a business question, get back a sourced, fact-checked report.

**Live demo:** _add your Streamlit Cloud URL here once deployed_

```
User's question
      |
Orchestrator agent (splits into sub-questions)
      |
   ------------------------------
   |          |          |
Research   Research   Research    (parallel, via a self-authored MCP server)
 agent 1    agent 2    agent 3
   |          |          |
   ------------------------------
      |
Combine findings (retries the whole batch once if everything came back empty)
      |
Fact-checker agent (cross-checks claims)
      |
Synthesis + human approval (you approve the outline)
      |
Writer agent (drafts final report)
      |
Final report (with citations)
```

## Why this exists

Most "AI agent" portfolio projects are a single LLM call with a system prompt.
This one is a real state machine (LangGraph) with:

- **Parallel specialist agents**, not a linear chain
- **A self-authored MCP server** — the research agents call tools over the Model
  Context Protocol rather than hitting APIs directly in agent code
- **A fact-checker** that flags single-sourced or contradictory claims
- **A human-in-the-loop approval gate** before the expensive drafting step runs
- **Retry and error handling on every LLM call and every research sub-question**,
  with a batch-level retry if a whole research round comes back empty, instead
  of crashing on the first transient API failure
- **An eval harness** that measures whether the fact-checker actually catches
  injected errors, instead of just "it seemed to work"

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your key into .env as GOOGLE_API_KEY=...
python main.py "Compare the top 3 CRM vendors for a 50-person startup"
```

Get a free key with no credit card at [Google AI Studio](https://aistudio.google.com).
This project runs entirely on Gemini's free tier (`gemini-3.6-flash`) — no
cost to build or demo. Note: free-tier traffic may be used by Google to
improve their models, so don't feed it confidential data.

Alternatively, run everything against a local model via [Ollama](https://ollama.com)
for a fully private, zero-network setup — swap `ChatGoogleGenerativeAI` for
`ChatOllama` in each agent file.

## Run the web UI locally

```bash
streamlit run streamlit_app.py
```

Same pipeline, same `.env` key, just a browser UI instead of the terminal.
This is also what's deployed live (see the link at the top).

## Run the eval suite

```bash
python -m eval.run_eval
```

**Latest result: 2/2 fact-checker test cases passed** — it correctly flagged
both a single-sourced claim and a direct contradiction between two sources.

## Project structure

```
agents/           one file per agent (orchestrator, research, fact_checker, synthesis, writer)
                   + utils.py (shared LLM retry/error handling)
mcp_servers/      self-authored MCP tool server
graph/            LangGraph state + graph wiring
eval/             eval test set + runner
main.py           CLI entry point
streamlit_app.py  web UI entry point (same pipeline, deployed on Streamlit Cloud)
```

## Status

- [x] Agent prompts + logic
- [x] Graph wiring with two interrupt points (research fan-out, human approval)
- [x] First MCP server (web search)
- [x] Retry/error handling on every LLM call and every research sub-question
- [x] Eval harness with injected-error test cases (2/2 passing)
- [x] Web UI (Streamlit) + free hosted deployment
- [ ] Docs/Drive MCP server (second parallel research source)
- [ ] Structured logging / tracing
