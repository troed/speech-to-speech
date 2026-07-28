import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MCP_BASE_URL = "http://localhost:8765"
MCP_ENDPOINT = "/mcp"


def _parse_sse_data(response_text: str) -> list[dict[str, Any]]:
    """Parse SSE event stream and return list of data payloads."""
    messages: list[dict[str, Any]] = []
    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            messages.append(payload)
    return messages


def _do_mcp_request(
    client: httpx.Client,
    body: dict[str, Any],
    session_id: str | None = None,
) -> httpx.Response:
    """Make an MCP JSON-RPC request and return the HTTP response."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id is not None:
        headers["MCP-Session-ID"] = session_id
    return client.post(MCP_ENDPOINT, headers=headers, json=body)


def _call_research(query: str) -> str:
    """Call tinysearch MCP ``research`` tool and return the answer prompt.

    Returns:
        The search-grounded answer prompt (raw text).

    Raises:
        RuntimeError: If the MCP call fails or returns an error.
    """
    with httpx.Client(base_url=MCP_BASE_URL, timeout=httpx.Timeout(90.0, connect=5.0)) as client:
        # Step 1: Initialize session
        init_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "speech-to-speech", "version": "1.0"},
            },
        }
        init_resp = _do_mcp_request(client, init_body)
        init_resp.raise_for_status()

        session_id = init_resp.headers.get("mcp-session-id")
        if not session_id:
            raise RuntimeError("MCP server did not return a session ID")

        init_data = _parse_sse_data(init_resp.text)
        if not init_data:
            raise RuntimeError("MCP initialize returned no data")
        init_result = init_data[0]
        if "error" in init_result:
            raise RuntimeError(f"MCP initialize failed: {init_result['error']}")

        # Step 2: Call research tool
        tool_body = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "research",
                "arguments": {"query": query},
            },
        }
        tool_resp = _do_mcp_request(client, tool_body, session_id=session_id)
        tool_resp.raise_for_status()

        tool_data = _parse_sse_data(tool_resp.text)
        if not tool_data:
            raise RuntimeError("MCP tools/call returned no data")
        tool_result = tool_data[0]
        if "error" in tool_result:
            raise RuntimeError(f"MCP tools/call failed: {tool_result['error']}")

        result = tool_result.get("result", {})
        if result.get("isError"):
            content = result.get("content", [])
            err_text = content[0].get("text", "Unknown error") if content else "Unknown error"
            raise RuntimeError(f"MCP research tool returned error: {err_text}")

        # Extract the answer from structuredContent or content[0].text
        structured = result.get("structuredContent", {})
        if structured and "answer" in structured:
            return structured["answer"]

        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            raw = content[0].get("text", "")
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "answer" in parsed:
                    return parsed["answer"]

        raise RuntimeError("MCP research tool returned unexpected response format")


def execute_web_search(query: str, num_results: int = 5) -> str:
    """Search the web via the existing tinysearch MCP server.

    Args:
        query: The search query string.
        num_results: Ignored (the research tool determines result count).

    Returns:
        A formatted search-results prompt, or an error message if the
        search fails.
    """
    try:
        return _call_research(query)
    except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("Web search via MCP failed: %s", exc)
        return f"Web search failed: unable to search for '{query}'."


SERVER_SIDE_TOOLS: dict[str, Any] = {
    "web_search": {
        "handler": execute_web_search,
        "definition": {
            "type": "function",
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
}


def is_server_side_tool(tool_name: str) -> bool:
    return tool_name in SERVER_SIDE_TOOLS


def get_tool_definition(tool_name: str) -> dict | None:
    entry = SERVER_SIDE_TOOLS.get(tool_name)
    return entry["definition"] if entry else None


def get_server_side_tool_definitions() -> list[dict]:
    return [entry["definition"] for entry in SERVER_SIDE_TOOLS.values()]


def set_mcp_base_url(url: str) -> None:
    global MCP_BASE_URL
    MCP_BASE_URL = url
