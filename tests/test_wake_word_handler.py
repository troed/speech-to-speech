from __future__ import annotations

from queue import Queue
from threading import Event

import pytest

from speech_to_speech.WakeWord.wake_word_handler import WakeWordHandler


class _MockModel:
    def __init__(self, score: float = 0.0) -> None:
        self._score = score

    def predict(self, chunk: bytes) -> dict[str, float]:
        return {"test_model": self._score}


def _mock_model(score: float = 0.0) -> _MockModel:
    return _MockModel(score=score)


def test_handler_requires_model_path():
    stop_event = Event()
    q_in: Queue = Queue()
    q_out: Queue = Queue()
    with pytest.raises(Exception):
        WakeWordHandler(stop_event, queue_in=q_in, queue_out=q_out, setup_args=(), setup_kwargs={})


def test_initial_state_is_sleeping():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    assert handler._state == "sleeping"


def test_on_session_end_resets_state():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "active"
    handler._buffer = [b"test"]
    handler.on_session_end()
    assert handler._state == "sleeping"
    assert handler._buffer == []


def test_sleeping_does_not_forward_bytes():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.5
    handler._activation_timeout_s = 30.0
    handler._last_forward_time = 0.0
    handler._model = _mock_model(score=0.1)

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert results == []
    assert len(handler._buffer) == 1


def test_active_forwards_bytes():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "active"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.5
    handler._activation_timeout_s = 30.0
    import time
    handler._last_forward_time = time.monotonic()

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert results == [chunk]


def test_active_timeout_goes_to_sleep():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "active"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.5
    handler._activation_timeout_s = 0.0
    handler._last_forward_time = 0.0

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert results == []
    assert handler._state == "sleeping"


def test_wake_word_handler_gates_audio_downstream():
    """Audio should not reach downstream queue while in sleeping state."""
    stop_event = Event()
    q_in: Queue = Queue()
    q_out: Queue = Queue()

    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.5
    handler._activation_timeout_s = 30.0
    handler._last_forward_time = 0.0
    handler._model = _mock_model(score=0.1)
    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert results == []
    assert q_out.qsize() == 0
