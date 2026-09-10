# Analyst-in-a-Box

A multi-agent research system: give it a business question, get back a sourced, fact-checked report.

**Live demo:** https://analyst-in-a-box-k4ahr2capppprwxqmstq7wkm.streamlit.app/

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
- **A self-authored MCP server** the research agents call tools over the Model
  Context Protocol (Tavily search) rather than hitting APIs directly in agent code
- **A fact-checker** that flags single-sourced or contradictory claims
- **A human-in-the-loop approval gate** before the expensive drafting step runs
- **Retry and error handling on every LLM call and every research sub-question**,
  including detecting a "successful but empty" search result and refusing to let
  the model fabricate placeholder citations for it, instead of crashing (or
  silently making things up) on the first transient failure
- **An eval harness** that measures whether the fact-checker actually catches
  injected errors, instead of just "it seemed to work"

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your keys into .env
python main.py "Compare the top 3 CRM vendors for a 50-person startup"
```

You'll need two free API keys, both added to `.env`:
- **`GOOGLE_API_KEY`** — the LLM. Get one at [Google AI Studio](https://aistudio.google.com),
  no credit card required. This project uses `gemini-flash-lite-latest`, chosen
  specifically for its much higher free-tier daily quota compared to the full
  Flash models. Note: free-tier traffic may be used by Google to improve their
  models, so don't feed it confidential data.
- **`TAVILY_API_KEY`** — web search for the research agents. Get one at
  [Tavily](https://tavily.com), no credit card required (1,000 searches/month
  free). Used instead of scraping DuckDuckGo directly, since DuckDuckGo (like
  most search engines) blocks datacenter/cloud IPs — which breaks search
  silently on any hosted deploy, not just this one.

Alternatively, run the LLM side against a local model via [Ollama](https://ollama.com)
for a fully private, zero-network setup — swap `ChatGoogleGenerativeAI` for
`ChatOllama` in each agent file.

## Run the web UI locally

```bash
streamlit run streamlit_app.py
```

Same pipeline, same `.env` keys, just a browser UI instead of the terminal.
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
mcp_servers/      self-authored MCP tool server (Tavily-backed web search)
graph/            LangGraph state + graph wiring
eval/             eval test set + runner
main.py           CLI entry point
streamlit_app.py  web UI entry point (same pipeline, deployed on Streamlit Cloud)
```

## Status

- [x] Agent prompts + logic
- [x] Graph wiring with two interrupt points (research fan-out, human approval)
- [x] MCP server for web search (Tavily — works reliably from hosted/cloud IPs)
- [x] Retry/error handling on every LLM call and every research sub-question,
      including rejecting empty search results instead of letting the model
      fabricate citations
- [x] Eval harness with injected-error test cases (2/2 passing)
- [x] Web UI (Streamlit) + free hosted deployment
- [ ] Docs/Drive MCP server (second parallel research source)
- [ ] Structured logging / tracing
