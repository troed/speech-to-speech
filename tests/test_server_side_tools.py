import json
from unittest.mock import patch

import httpx
import pytest

from speech_to_speech.LLM.server_side_tools import (
    _parse_sse_data,
    execute_web_search,
    get_server_side_tool_definitions,
    get_tool_definition,
    is_server_side_tool,
)

MOCK_ANSWER = """SEARCH-GROUNDED ANSWER PROMPT

QUESTION
test query

RESULTS
RESULT 1
Some test result (https://example.com)
"""


def _make_mcp_success_response(answer: str) -> str:
    payload = json.dumps({"answer": answer})
    data = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": payload}],
                "structuredContent": {"answer": answer},
                "isError": False,
            },
        }
    )
    return f"event: message\ndata: {data}\n\n"


class TestParseSseData:
    def test_parses_single_event(self):
        raw = 'event: message\ndata: {"key": "value"}\n\n'
        result = _parse_sse_data(raw)
        assert result == [{"key": "value"}]

    def test_parses_multiple_events(self):
        raw = 'event: message\ndata: {"a": 1}\n\nevent: message\ndata: {"b": 2}\n\n'
        result = _parse_sse_data(raw)
        assert result == [{"a": 1}, {"b": 2}]

    def test_ignores_non_data_lines(self):
        raw = 'event: message\ndata: {"key": "value"}\n\n:comment\n'
        result = _parse_sse_data(raw)
        assert result == [{"key": "value"}]


class TestIsServerSideTool:
    def test_returns_true_for_web_search(self):
        assert is_server_side_tool("web_search") is True

    def test_returns_false_for_unknown(self):
        assert is_server_side_tool("get_weather") is False


class TestGetToolDefinition:
    def test_returns_dict(self):
        definition = get_tool_definition("web_search")
        assert definition is not None
        assert definition["name"] == "web_search"
        assert "parameters" in definition

    def test_returns_none_for_unknown(self):
        assert get_tool_definition("unknown_tool") is None

    def test_has_query_required(self):
        definition = get_tool_definition("web_search")
        assert "query" in definition["parameters"]["required"]


class TestGetServerSideToolDefinitions:
    def test_returns_list_of_definitions(self):
        definitions = get_server_side_tool_definitions()
        assert len(definitions) >= 1
        web_search = definitions[0]
        assert web_search["type"] == "function"
        assert web_search["name"] == "web_search"
        assert "parameters" in web_search

    def test_valid_openai_function_schema(self):
        web_search = get_server_side_tool_definitions()[0]
        assert web_search["type"] == "function"
        assert web_search["name"] == "web_search"
        assert "description" in web_search
        assert web_search["parameters"]["type"] == "object"
        assert "properties" in web_search["parameters"]
        assert "required" in web_search["parameters"]


class TestExecuteWebSearch:
    def test_returns_answer_from_mcp(self):
        with patch(
            "speech_to_speech.LLM.server_side_tools._call_research",
            return_value=MOCK_ANSWER,
        ):
            result = execute_web_search("test query")
            assert "SEARCH-GROUNDED ANSWER PROMPT" in result
            assert "test query" in result

    def test_mcp_failure_returns_error_message(self):
        with patch(
            "speech_to_speech.LLM.server_side_tools._call_research",
            side_effect=RuntimeError("MCP error"),
        ):
            result = execute_web_search("broken query")
            assert "Web search failed" in result
            assert "broken query" in result

    def test_num_results_is_accepted_but_ignored(self):
        with patch(
            "speech_to_speech.LLM.server_side_tools._call_research",
            return_value=MOCK_ANSWER,
        ):
            result = execute_web_search("test query", num_results=3)
            assert "SEARCH-GROUNDED ANSWER PROMPT" in result
