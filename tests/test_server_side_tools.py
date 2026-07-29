from speech_to_speech.LLM.server_side_tools import is_server_side_tool


def test_is_server_side_tool_returns_false_for_unknown():
    assert is_server_side_tool("nonexistent_tool") is False
