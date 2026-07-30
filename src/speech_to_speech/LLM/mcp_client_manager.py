from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mcp
from mcp_types import TextContent

logger = logging.getLogger(__name__)


def _exc_message(exc: BaseException) -> str:
    """Extract a readable message from an exception, unwrapping ExceptionGroups."""
    if isinstance(exc, BaseExceptionGroup):
        inner = exc.exceptions[0] if exc.exceptions else exc
        return _exc_message(inner)
    return f"{type(exc).__name__}: {exc}"


class MCPClientManager:
    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._servers: dict[str, dict[str, Any]] = {}
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

    def _build_http_client(self, cfg: dict[str, Any]) -> mcp.Client:
        """Build an MCP Client for an HTTP server, optionally with custom headers."""
        headers = cfg.get("headers")
        if headers:
            from mcp.client.streamable_http import streamable_http_client

            import httpx2

            http_client = httpx2.AsyncClient(headers=headers)
            transport = streamable_http_client(cfg["url"], http_client=http_client)
            return mcp.Client(transport)
        return mcp.Client(cfg["url"])

    async def start(self) -> None:
        """Connect to all configured servers, discover tools, then close connections."""
        import mcp
        from mcp.client.stdio import StdioServerParameters, stdio_client

        for name, cfg in self._servers.items():
            transport_type = cfg["type"]
            try:
                if transport_type == "http":
                    client = self._build_http_client(cfg)
                else:
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env"),
                    )
                    client = mcp.Client(stdio_client(params))
            except Exception as exc:
                logger.warning("MCP server '%s': failed to create client: %s", name, _exc_message(exc))
                continue

            try:
                async with client:
                    try:
                        tools_result = await client.list_tools()
                    except Exception as exc:
                        logger.warning("MCP server '%s': tools/list failed: %s", name, _exc_message(exc))
                        continue

                    tool_overrides = cfg.get("toolOverrides", {})
                    for tool in tools_result.tools:
                        overrides = tool_overrides.get(tool.name, {})
                        description = overrides.get("description", tool.description or "")
                        parameters = tool.input_schema if tool.input_schema else {"type": "object", "properties": {}}
                        prop_overrides = overrides.get("properties", {})
                        if prop_overrides:
                            props = dict(parameters.get("properties", {}))
                            for prop_name, prop_cfg in prop_overrides.items():
                                existing = dict(props.get(prop_name, {}))
                                existing.update(prop_cfg)
                                props[prop_name] = existing
                            parameters = {**parameters, "properties": props}
                        definition: dict[str, Any] = {
                            "type": "function",
                            "name": tool.name,
                            "description": description,
                            "parameters": parameters,
                        }
                        self._register_tool(tool.name, name, definition)

                    logger.info(
                        "MCP server '%s': %d tools discovered",
                        name,
                        len(tools_result.tools),
                    )
            except Exception as exc:
                logger.warning("MCP server '%s': connection failed: %s", name, _exc_message(exc))

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        import asyncio
        import mcp
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_name = self._tool_to_server.get(tool_name)
        if server_name is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        cfg = self._servers.get(server_name)
        if cfg is None:
            raise RuntimeError(f"No server config for '{server_name}'")

        transport_type = cfg["type"]
        if transport_type == "http":
            client = self._build_http_client(cfg)
        else:
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env"),
            )
            client = mcp.Client(stdio_client(params))

        async def _run() -> str:
            async with client:
                result = await client.call_tool(tool_name, arguments)
            return "\n".join(
                block.text
                for block in result.content
                if isinstance(block, TextContent)
            )

        return asyncio.run(_run())

    @property
    def server_count(self) -> int:
        return len(self._servers)
