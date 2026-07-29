from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mcp
from mcp_types import TextContent

logger = logging.getLogger(__name__)


class MCPClientManager:
    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._servers: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, mcp.Client] = {}
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

    def _register_tool(
        self, tool_name: str, server_name: str, definition: dict[str, Any]
    ) -> None:
        """Register a tool and its definition, handling namespace collisions."""
        if tool_name in self._tool_to_server:
            existing = self._tool_to_server[tool_name]
            logger.warning(
                "Tool '%s' from server '%s' already registered from server '%s'",
                tool_name,
                server_name,
                existing,
            )
            for i, d in enumerate(self._tool_definitions):
                if d["name"] == tool_name:
                    self._tool_definitions[i] = definition
                    break
        else:
            self._tool_definitions.append(definition)
        self._tool_to_server[tool_name] = server_name

    def is_server_side_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_to_server

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return list(self._tool_definitions)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        import asyncio
        server_name = self._tool_to_server.get(tool_name)
        if server_name is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        client = self._clients.get(server_name)
        if client is None:
            raise RuntimeError(f"No client for server '{server_name}'")
        result = asyncio.run(client.call_tool(tool_name, arguments))
        return "\n".join(
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        )

    async def start(self) -> None:
        """Connect to all configured servers and discover tools."""
        from mcp.client.stdio import StdioServerParameters, stdio_client

        for name, cfg in self._servers.items():
            transport_type = cfg["type"]
            try:
                if transport_type == "http":
                    client = mcp.Client(cfg["url"])
                else:
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env"),
                    )
                    transport = stdio_client(params)
                    client = mcp.Client(transport)
            except Exception as exc:
                logger.warning("MCP server '%s': failed to create client: %s", name, exc)
                continue

            try:
                await client.__aenter__()
                self._clients[name] = client
            except Exception as exc:
                logger.warning("MCP server '%s': failed to connect: %s", name, exc)
                continue

            try:
                tools_result = await client.list_tools()
            except Exception as exc:
                logger.warning("MCP server '%s': tools/list failed: %s", name, exc)
                continue

            for tool in tools_result.tools:
                definition: dict[str, Any] = {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema if tool.input_schema else {"type": "object", "properties": {}},
                }
                self._register_tool(tool.name, name, definition)

            logger.info(
                "MCP server '%s': connected, %d tools discovered",
                name,
                len(tools_result.tools),
            )

    async def close(self) -> None:
        """Close all client connections."""
        for name, client in list(self._clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("MCP server '%s': error closing: %s", name, exc)
        self._clients.clear()

    @property
    def server_count(self) -> int:
        return len(self._servers)
