from __future__ import annotations

import time
from queue import Queue
from threading import Event

import numpy as np
import pytest

from speech_to_speech.WakeWord.wake_word_handler import WakeWordHandler

CHIME_TEST_BYTES = b"\x00\x01\x02\x03"


class _MockModel:
    def __init__(self, score: float = 0.0) -> None:
        self._score = score

    def predict(self, audio: np.ndarray) -> dict[str, float]:
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
    handler._activation_start_time = 0.0
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
    handler._activation_start_time = time.monotonic()

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
    handler._activation_start_time = 0.0

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
    handler._activation_start_time = 0.0
    handler._model = _mock_model(score=0.1)
    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert results == []
    assert q_out.qsize() == 0


def test_wake_handler_plays_chime_on_detection():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.0
    handler._activation_timeout_s = 30.0
    handler._activation_start_time = 0.0
    handler._cooldown_until = 0.0
    handler._model = _mock_model(score=0.9)
    handler._should_listen = None
    handler._wake_chime_bytes = CHIME_TEST_BYTES
    chime_queue: Queue = Queue()
    handler._chime_output_queue = chime_queue

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert len(results) == 0  # no buffer or wakeword forwarded
    assert chime_queue.qsize() == 1
    assert chime_queue.get_nowait() == CHIME_TEST_BYTES


def test_wake_handler_no_chime_when_not_configured():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.0
    handler._activation_timeout_s = 30.0
    handler._activation_start_time = 0.0
    handler._cooldown_until = 0.0
    handler._model = _mock_model(score=0.9)
    handler._should_listen = None
    handler._wake_chime_bytes = None
    handler._chime_output_queue = None

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert len(results) == 0


def test_wake_handler_chime_with_buffered_audio():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = [b"\xaa" * 1024, b"\xbb" * 1024]
    handler._preroll_chunks = 5
    handler._threshold = 0.0
    handler._activation_timeout_s = 30.0
    handler._activation_start_time = 0.0
    handler._cooldown_until = 0.0
    handler._model = _mock_model(score=0.9)
    handler._should_listen = None
    handler._wake_chime_bytes = CHIME_TEST_BYTES
    chime_queue: Queue = Queue()
    handler._chime_output_queue = chime_queue

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert len(results) == 0  # nothing forwarded on wake
    assert chime_queue.qsize() == 1
