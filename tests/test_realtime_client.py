import base64

from speech_to_speech.client import (
    _build_session_update,
    _decode_output_audio,
    _encode_input_audio,
    _parse_realtime_text_event,
)


class TestEncodeInputAudio:
    def test_encodes_pcm_bytes_as_base64_append_event(self):
        chunk = b"\x00\x01\x02\x03"

        result = _encode_input_audio(chunk)

        assert result["type"] == "input_audio_buffer.append"
        assert result["audio"] == base64.b64encode(chunk).decode("ascii")

    def test_encodes_empty_chunk(self):
        result = _encode_input_audio(b"")

        assert result["type"] == "input_audio_buffer.append"
        assert result["audio"] == ""


class TestDecodeOutputAudio:
    def test_decodes_base64_delta_to_pcm_bytes(self):
        pcm = b"\x00\x01\x02\x03"
        b64 = base64.b64encode(pcm).decode("ascii")
        event = {"type": "response.output_audio.delta", "delta": b64}

        result = _decode_output_audio(event)

        assert result == pcm

    def test_returns_none_for_non_audio_event(self):
        event = {"type": "response.done", "response": {}}

        result = _decode_output_audio(event)

        assert result is None

    def test_returns_none_for_missing_delta(self):
        event = {"type": "response.output_audio.delta"}

        result = _decode_output_audio(event)

        assert result is None


class TestParseRealtimeTextEvent:
    def test_parses_user_transcription_delta(self):
        event = {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": "Hello wor",
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "partial_transcription", "delta": "Hello wor"}

    def test_parses_user_transcription_completed(self):
        event = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hello world",
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "transcription_completed", "transcript": "Hello world"}

    def test_parses_speech_started(self):
        event = {"type": "input_audio_buffer.speech_started"}

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "speech_started"}

    def test_parses_assistant_text(self):
        event = {
            "type": "response.output_audio_transcript.done",
            "transcript": "Greetings, Captain.",
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "assistant_text", "text": "Greetings, Captain."}

    def test_parses_response_failed(self):
        event = {
            "type": "response.done",
            "response": {
                "status": "failed",
                "status_details": {"error": {"message": "Something broke"}},
            },
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "response_failed", "error": "Something broke"}

    def test_parses_token_usage_in_response_done(self):
        event = {
            "type": "response.done",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 42, "output_tokens": 7},
            },
        }

        result = _parse_realtime_text_event(event)

        assert result == {
            "kind": "response_done",
            "input_tokens": 42,
            "output_tokens": 7,
        }

    def test_returns_response_done_for_completed_without_usage(self):
        event = {
            "type": "response.done",
            "response": {"status": "completed"},
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "response_done"}

    def test_returns_none_for_failed_without_error(self):
        event = {
            "type": "response.done",
            "response": {"status": "failed"},
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "response_failed", "error": "Unknown error"}

    def test_returns_response_done_for_cancelled(self):
        event = {
            "type": "response.done",
            "response": {"status": "cancelled"},
        }

        result = _parse_realtime_text_event(event)

        assert result == {"kind": "response_done"}

    def test_returns_none_for_non_dict_input(self):
        result = _parse_realtime_text_event("not a dict")
        assert result is None


class TestBuildSessionUpdate:
    def test_builds_minimal_session_update(self):
        result = _build_session_update()

        assert result["type"] == "session.update"
        assert "session" in result

    def test_audio_output_format_is_16khz_pcm(self):
        result = _build_session_update()
        audio = result["session"]["audio"]

        assert audio["output"]["format"]["type"] == "pcm16"
        assert audio["output"]["format"]["rate"] == 16000

    def test_output_modalities_includes_audio_and_text(self):
        result = _build_session_update()

        assert "audio" in result["session"]["output_modalities"]
        assert "text" in result["session"]["output_modalities"]

    def test_turn_detection_is_server_vad(self):
        result = _build_session_update()

        assert result["session"]["turn_detection"]["type"] == "server_vad"

    def test_accepts_optional_voice_and_instructions(self):
        result = _build_session_update(voice="computer", instructions="Be a starship computer.")

        assert result["session"]["voice"] == "computer"
        assert result["session"]["instructions"] == "Be a starship computer."

    def test_no_optional_fields_when_omitted(self):
        result = _build_session_update()

        assert "voice" not in result["session"]
        assert "instructions" not in result["session"]
