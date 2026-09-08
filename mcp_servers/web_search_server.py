"""
MCP server exposing a single tool: web_search.

This is the piece that makes the project "MCP-native" rather than just
calling an API directly inside the agent code. Any MCP-compatible client
(LangGraph via an MCP adapter, Claude Desktop, etc.) can call this tool
without knowing anything about the underlying search provider -- that's
the point of the protocol: the tool's implementation is decoupled from
the agent that uses it.

WHY TAVILY, NOT DUCKDUCKGO: the original version of this file used the
duckduckgo-search library. It works fine on a home/laptop connection,
but DuckDuckGo (like Google, Bing, and Brave) actively blocks requests
from datacenter/cloud IP ranges to stop scraping -- and every hosted
platform (Streamlit Cloud, Render, Railway, etc.) runs on exactly that
kind of IP. The result on a deployed demo: search calls silently return
zero results instead of erroring, which -- left unhandled -- lets the
LLM fabricate plausible-looking fake citations (see agents/research.py
for how that failure mode is now caught). Tavily is a search API built
specifically for AI agents, authenticates with an API key rather than
scraping HTML, and works identically from a laptop or a cloud host.
Free tier: 1,000 searches/month, no credit card, from tavily.com.

Run standalone for testing:
    python mcp_servers/web_search_server.py
"""

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

load_dotenv()  # so TAVILY_API_KEY from .env is available when run standalone

mcp = FastMCP("web-search-server")

_api_key = os.environ.get("TAVILY_API_KEY")
_client = TavilyClient(api_key=_api_key) if _api_key else None


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for a query and return a list of results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of dicts, each with 'title', 'href', and 'body' keys.
        (Field names kept identical to the old DuckDuckGo version so
        agents/research.py didn't need to change when the provider did.)
    """
    if _client is None:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key (no credit card) at "
            "https://tavily.com and add it to .env (local) or your "
            "platform's secrets (deployed)."
        )

    response = _client.search(query, max_results=max_results, search_depth="basic")
    return [
        {
            "title": r.get("title", ""),
            "href": r.get("url", ""),
            "body": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


if __name__ == "__main__":
    mcp.run()