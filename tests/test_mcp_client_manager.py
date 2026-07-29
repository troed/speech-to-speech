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
