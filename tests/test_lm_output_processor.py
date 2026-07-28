from queue import Queue
from threading import Event, Thread
from unittest.mock import patch

from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def _make_tool(*, name: str, arguments: str, call_id: str = "call_1") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="1",
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
        type="function_call",
    )


def _processor(tracker: SpeculativeTurnTracker) -> LMOutputProcessor:
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=Queue(), speculative_turns=tracker)
    return processor


def test_stale_end_of_response_is_not_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    processor = _processor(tracker)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))

    assert outputs == []


def test_latest_end_of_response_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    processor = _processor(tracker)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=1)))

    assert len(outputs) == 1
    assert outputs[0].turn_id == "turn_1"
    assert outputs[0].turn_revision == 1


def test_cancel_generation_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        )
    )

    assert len(outputs) == 1
    assert outputs[0].cancel_generation == 7


def test_text_only_chunk_is_not_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                response=RealtimeResponseCreateParams(output_modalities=["text"]),
            )
        )
    )

    assert outputs == []
    # The assistant text still reaches clients even when TTS is skipped.
    event = processor.text_output_queue.get_nowait()
    assert event.text == "hello"


def test_audio_chunk_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                response=RealtimeResponseCreateParams(output_modalities=["audio"]),
            )
        )
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], TTSInput)
    assert outputs[0].text == "hello"


def test_empty_modalities_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                response=RealtimeResponseCreateParams(output_modalities=[]),
            )
        )
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], TTSInput)


def test_pending_reopen_holds_assistant_chunk_until_cancelled():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    processor = _processor(tracker)
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(
            processor.process(
                LLMResponseChunk(
                    text="hello",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_processor)
    thread.start()

    assert not done.wait(0.05)
    tracker.cancel_reopen_candidate("turn_1", candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert len(outputs) == 1
    assert outputs[0].text == "hello"
    event = processor.text_output_queue.get_nowait()
    assert event.text == "hello"


def test_reopen_grace_holds_assistant_chunk_until_elapsed():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.08)
    processor = _processor(tracker)
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(
            processor.process(
                LLMResponseChunk(
                    text="hello",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_processor)
    thread.start()

    assert not done.wait(0.02)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert len(outputs) == 1
    assert outputs[0].text == "hello"
    event = processor.text_output_queue.get_nowait()
    assert event.text == "hello"


def test_confirmed_reopen_drops_held_assistant_chunk():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    processor = _processor(tracker)
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(
            processor.process(
                LLMResponseChunk(
                    text="hello",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_processor)
    thread.start()

    assert not done.wait(0.05)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []
    assert processor.text_output_queue.empty()


def test_server_side_tools_filtered_from_client_event():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    with patch("speech_to_speech.LLM.lm_output_processor.is_server_side_tool", return_value=True):
        chunk = LLMResponseChunk(
            text="",
            turn_id="turn_1",
            turn_revision=0,
            tools=[
                _make_tool(name="web_search", arguments='{"query": "test"}'),
            ],
        )
        outputs = list(processor.process(chunk))
        assert outputs == []
        event = processor.text_output_queue.get_nowait()
        assert event.text == ""
        assert event.tools == []  # all tools filtered out


def test_client_tools_passed_through():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    with patch("speech_to_speech.LLM.lm_output_processor.is_server_side_tool", return_value=False):
        chunk = LLMResponseChunk(
            text="hello",
            turn_id="turn_1",
            turn_revision=0,
            tools=[
                _make_tool(name="get_weather", arguments='{"city": "London"}'),
            ],
        )
        outputs = list(processor.process(chunk))
        event = processor.text_output_queue.get_nowait()
        assert event.text == "hello"
        assert len(event.tools) == 1
        assert event.tools[0].name == "get_weather"


def test_mixed_tools_filtered_correctly():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    def fake_is_server_side(name: str) -> bool:
        return name in ("web_search",)

    with patch("speech_to_speech.LLM.lm_output_processor.is_server_side_tool", side_effect=fake_is_server_side):
        chunk = LLMResponseChunk(
            text="checking...",
            turn_id="turn_1",
            turn_revision=0,
            tools=[
                _make_tool(name="web_search", arguments='{"query": "test"}'),
                _make_tool(name="get_weather", arguments='{"city": "London"}'),
            ],
        )
        outputs = list(processor.process(chunk))
        event = processor.text_output_queue.get_nowait()
        assert event.text == "checking..."
        assert event.tools is not None
        assert len(event.tools) == 1
        assert event.tools[0].name == "get_weather"
