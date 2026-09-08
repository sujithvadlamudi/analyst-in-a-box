"""
MCP server exposing a single tool: web_search.

This is the piece that makes the project "MCP-native" rather than just
calling an API directly inside the agent code. Any MCP-compatible client
(LangGraph via an MCP adapter, Claude Desktop, etc.) can call this tool
without knowing anything about DuckDuckGo internally -- that's the point
of the protocol: the tool's implementation is decoupled from the agent
that uses it.

Run standalone for testing:
    python mcp_servers/web_search_server.py
"""

from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("web-search-server")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for a query and return a list of results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of dicts, each with 'title', 'href', and 'body' keys.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


if __name__ == "__main__":
    mcp.run()
