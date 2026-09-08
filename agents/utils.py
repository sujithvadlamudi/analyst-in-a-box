"""
Shared error-handling helper for every agent node.

Every agent in this project calls out to the Gemini API. Those calls can
fail transiently (rate limits on the free tier, a dropped connection,
a timeout) -- and until this file existed, none of the agents caught
that. A single hiccup on any node would crash the whole graph run with
a raw traceback and no way to recover.

`call_llm` wraps every LLM call with:
  - up to `max_retries` attempts, with a short backoff between them
  - a clear, labeled error if all attempts are exhausted
  - a print statement on each retry, so failures are visible in the
    terminal instead of happening silently
"""

import time


class AgentError(RuntimeError):
    """Raised when an agent node exhausts its retries. Caught in
    main.py so the program can exit cleanly instead of with a raw
    traceback."""


def get_text(response) -> str:
    """Extract plain text from an LLM response, regardless of format.

    Gemini 2.x models return response.content as a plain string.
    Gemini 3+ models return it as a list of content blocks instead,
    e.g. [{"type": "text", "text": "..."}]. Every agent used to call
    response.content.strip() directly, which crashed with
    AttributeError: 'list' object has no attribute 'strip' the moment
    the model was upgraded to gemini-3.6-flash.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect a 429/rate-limit error across different exception classes
    and library versions, by checking the message rather than relying
    on a specific exception type (which varies between google-genai
    SDK versions)."""
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "RATE LIMIT" in text


def call_llm(llm, messages: list[dict], node_name: str, max_retries: int = 2):
    last_error = None
    for attempt in range(1, max_retries + 2):  # e.g. max_retries=2 -> 3 tries total
        try:
            return llm.invoke(messages)
        except Exception as exc:  # broad on purpose: API/network errors vary a lot
            last_error = exc
            if attempt <= max_retries:
                if _is_rate_limit_error(exc):
                    # The free tier is roughly 10-15 requests/minute.
                    # A short 1s/2s backoff just re-hits the same
                    # per-minute window and gets 429'd again immediately.
                    # Wait long enough for the rolling window to clear.
                    wait = 20 * attempt  # 20s, 40s...
                    reason = "rate limited (free tier)"
                else:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    reason = exc.__class__.__name__
                print(
                    f"  [{node_name}] call failed ({reason}), "
                    f"retrying in {wait}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait)
    raise AgentError(
        f"{node_name} failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error