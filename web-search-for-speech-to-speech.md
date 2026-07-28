# Adding server-side web search to HuggingFace speech-to-speech

## Architecture

```
Restricted computer (voice agent client, Realtime WebSocket)
    |
    | WebSocket (OpenAI Realtime protocol)
    v
speech-to-speech server  ──chat-completions/responses──►  llama-server
    |
    | (proposed) SearXNG self-hosted search
    v
SearXNG (localhost:4000 or wherever)
```

The restricted client runs the HuggingFace speech-to-speech app and connects to the
speech-to-speech server via the OpenAI Realtime WebSocket protocol. The server talks
to llama.cpp's llama-server via `/v1/chat/completions` or `/v1/responses`.

## Problem

The speech-to-speech server supports tool/function calling (the LLM can request tool
calls), but it forwards ALL tool calls to the WebSocket client. The restricted client
cannot execute tools (no MCP, no HTTP client capability). So web search requests from
the model are sent to the client and dropped.

## Speech-to-speech tool call flow (current)

1. Client sends `session.update` with `tools` defined (incl. `web_search`)
2. Client sends audio -> STT -> LLM
3. LLM returns tool calls (e.g., `web_search(query="...")`)
4. Server's `response.py` handler sends `response.function_call_arguments.done`
   to the client over WebSocket
5. (dead end -- client cannot execute the tool)

## Proposed change: server-side tool execution hook

Add a hook in the LLM output processing pipeline that intercepts tool calls
before they reach the WebSocket client. If the tool is a server-side tool
(like `web_search`), the server executes it, feeds the result back to the LLM,
and returns the final response to the client. The client never sees the tool call.

### Modified flow

1. Client sends audio -> STT -> LLM
2. LLM returns tool calls (e.g., `web_search(query="...")`)
3. Server detects `web_search` is a server-side tool
4. Server queries SearXNG at `http://localhost:4000/search?q=...&format=json`
5. Server injects `function_call_output` into the chat history
6. Server re-triggers the LLM with the tool result included
7. LLM generates final response with web search results
8. Server streams final response as audio to the client
9. Client hears the answer -- no tool calls ever reach it

### Key code locations to modify

All paths relative to the speech-to-speech repo root.

#### 1. Where tool calls from the LLM are received

**`src/speech_to_speech/LLM/lm_output_processor.py`**

The `LMOutputProcessor.process()` method receives `LLMResponseChunk` objects.
When `lm_output.tools` is non-empty, they are attached to `AssistantTextEvent`
and forwarded to the client. This is where we intercept.

```python
# Current (line ~45-60):
if lm_output.tools:
    event.tools = lm_output.tools
    logger.info(f"Sending to clients: tools={[t.name for t in lm_output.tools]}")
```

We add a check: if any tool is server-side (e.g., `web_search`), route it to
a new `ServerSideToolExecutor` instead of forwarding it to the client.

#### 2. Where tool calls are sent to the WebSocket client

**`src/speech_to_speech/api/openai_realtime/handlers/response.py`**

The `ResponseHandler.on_assistant_text()` converts tool calls to
`response.function_call_arguments.done` WebSocket events. If we intercept
at the `lm_output_processor.py` level, the tool calls never reach this handler.

However, there's a fallback path to also check: if a tool call somehow reaches
the response handler, we should skip it if it's a server-side tool.

#### 3. Where the LLM response is re-triggered

**`src/speech_to_speech/LLM/base_openai_compatible_language_model.py`**

The `BaseOpenAICompatibleHandler.process()` method is the main LLM call.
We need to:

1. After detecting a server-side tool call, inject the `function_call_output`
   into the chat (using `chat.append_tool_output()` from `chat.py`)
2. Call `process()` again with the updated chat to get the LLM's final response

This means the server-side tool execution needs to happen as a loop around
`process()` -- call LLM -> tool call? -> execute tool -> call LLM again -> done.

#### 4. Chat history management for tool results

**`src/speech_to_speech/LLM/chat.py`**

`Chat.append_tool_output(call_id, output)` -- appends a `function_call_output`
item and re-injects the paired `function_call` if it was evicted during
history compaction. Already works; just needs to be called.

#### 5. Where tools are defined/registered

**`src/speech_to_speech/api/openai_realtime/handlers/session.py`**

Session handler where `session.update` events are processed. The client
sends tool definitions here. Need to decide: do we want the client to
declare `web_search`, or do we add it server-side?

Simpler approach: add `web_search` tool definition server-side so the client
doesn't need to know about it. The server injects it into the tools list
before sending them to the LLM.

## Implementation sketch

### New file: `src/speech_to_speech/LLM/server_side_tools.py`

```python
"""
Handles tools that are executed server-side (e.g., web search)
rather than forwarded to the WebSocket client.
"""

import httpx
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SEARXNG_BASE_URL = "http://localhost:4000"  # configurable


async def execute_web_search(query: str, num_results: int = 5) -> str:
    """Query SearXNG and return formatted results."""
    params = {
        "q": query,
        "format": "json",
        "number_of_results": num_results,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SEARXNG_BASE_URL}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return f"No search results found for '{query}'."

    lines = [f"Web search results for '{query}':\n"]
    for r in results[:num_results]:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"- {title}: {snippet} ({url})")

    return "\n".join(lines)


# Registry of server-side tools with their execution functions
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
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
}


def is_server_side_tool(tool_name: str) -> bool:
    return tool_name in SERVER_SIDE_TOOLS


def get_tool_definition(tool_name: str) -> dict | None:
    entry = SERVER_SIDE_TOOLS.get(tool_name)
    return entry["definition"] if entry else None
```

### Modified: `src/speech_to_speech/LLM/lm_output_processor.py`

Add a new class `ServerSideToolOrchestrator` that wraps the LLM handler
and auto-executes server-side tools:

```python
# Pseudo-code for the orchestration:
class ServerSideToolOrchestrator:
    def __init__(self, llm_handler, chat):
        self.llm = llm_handler
        self.chat = chat

    async def process_with_server_tools(self, llm_input):
        while True:
            result = await self.llm.process(llm_input)
            if result.tools and all(is_server_side_tool(t.name) for t in result.tools):
                for tool in result.tools:
                    output = await execute_web_search(**json.loads(tool.arguments))
                    self.chat.append_tool_output(tool.call_id, output)
                # Re-trigger LLM with tool results
                llm_input = self.chat.to_llm_input()
            else:
                return result
```

## Configuration

Add CLI flags to `speech-to-speech`:

- `--server-tools`: comma-separated list of tools to execute server-side
  (e.g., `--server-tools web_search`)
- `--searxng-url`: SearXNG API URL (default: `http://localhost:4000`)

These would go in the module arguments or a new arguments class.

## SearXNG setup (already planned)

SearXNG exposes a JSON API at `/search?q=...&format=json`. Run it in Docker:

```yaml
# docker-compose.yml snippet
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "4000:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:4000
    volumes:
      - ./searxng-config:/etc/searxng:ro
```

The speech-to-speech server queries SearXNG at `http://localhost:4000/search`
when the model requests `web_search`.

## Security notes

- SearXNG should be bound to localhost only (not exposed to the network)
- `--server-tools` should have appropriate warnings about executing external
  API calls from the server
- SearXNG does NOT share the user's IP with search engines (privacy-preserving)
