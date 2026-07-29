from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp import Client
from mcp_types import TextContent

logger = logging.getLogger(__name__)


class MCPClientManager:
    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._servers: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, Client] = {}
        self._tool_to_server: dict[str, str] = {}
        self._tool_definitions: list[dict[str, Any]] = []
        self._parse_config()

    def _parse_config(self) -> None:
        try:
            raw = json.loads(self._config_path.read_text())
        except FileNotFoundError:
            logger.warning("MCP config file not found: %s", self._config_path)
            return
        except json.JSONDecodeError as exc:
            logger.warning("MCP config file is invalid JSON: %s", exc)
            return

        servers = raw.get("mcpServers", {})
        for name, cfg in servers.items():
            transport_type = cfg.get("type")
            if transport_type not in ("stdio", "http"):
                logger.warning("MCP server '%s': unknown type '%s', skipping", name, transport_type)
                continue
            if transport_type == "http" and "url" not in cfg:
                logger.warning("MCP server '%s': http type requires 'url', skipping", name)
                continue
            if transport_type == "stdio" and "command" not in cfg:
                logger.warning("MCP server '%s': stdio type requires 'command', skipping", name)
                continue
            self._servers[name] = cfg

    @property
    def server_count(self) -> int:
        return len(self._servers)
