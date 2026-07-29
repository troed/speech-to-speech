from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from speech_to_speech.LLM.mcp_client_manager import MCPClientManager


def test_parses_mcp_json_with_stdio_and_http_servers():
    config = {
        "mcpServers": {
            "tinysearch": {
                "type": "http",
                "url": "http://localhost:8765/mcp",
            },
            "filesystem": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@mcp/server-filesystem"],
                "env": {"HOME": "/tmp"},
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        path = f.name

    try:
        manager = MCPClientManager(path)
        assert len(manager._servers) == 2
        assert manager._servers["tinysearch"]["type"] == "http"
        assert manager._servers["tinysearch"]["url"] == "http://localhost:8765/mcp"
        assert manager._servers["filesystem"]["type"] == "stdio"
        assert manager._servers["filesystem"]["command"] == "npx"
    finally:
        Path(path).unlink()


def test_is_server_side_tool():
    manager = MCPClientManager.__new__(MCPClientManager)
    manager._tool_to_server = {"web_search": "tinysearch", "read_file": "filesystem"}
    manager._tool_definitions = []

    assert manager.is_server_side_tool("web_search") is True
    assert manager.is_server_side_tool("read_file") is True
    assert manager.is_server_side_tool("unknown_tool") is False


def test_get_tool_definitions_returns_openai_compatible_schema():
    manager = MCPClientManager.__new__(MCPClientManager)
    manager._tool_definitions = [
        {
            "type": "function",
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    ]
    manager._tool_to_server = {"web_search": "tinysearch"}

    defs = manager.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "web_search"
    assert defs[0]["type"] == "function"
    assert "parameters" in defs[0]


def test_namespace_collision_warns():
    import logging
    from io import StringIO

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("speech_to_speech.LLM.mcp_client_manager")
    logger.addHandler(handler)

    manager = MCPClientManager.__new__(MCPClientManager)
    manager._tool_to_server = {}
    manager._tool_definitions = []

    manager._register_tool("weather", "server_a", {"type": "function", "name": "weather"})
    manager._register_tool("weather", "server_b", {"type": "function", "name": "weather"})

    log_output = log_stream.getvalue()
    assert "already registered" in log_output
    assert manager._tool_to_server["weather"] == "server_b"

    logger.removeHandler(handler)
