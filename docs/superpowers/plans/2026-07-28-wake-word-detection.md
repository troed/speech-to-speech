# Wake Word Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional "computer" wake word detection to the speech-to-speech pipeline using openWakeWord.

**Architecture:** A `WakeWordHandler` (BaseHandler[VADIn, VADIn]) sits between audio input and VAD. It runs openWakeWord on incoming audio frames. When the wake word is detected, it forwards buffered + live audio to VAD. After a configurable timeout of silence, it goes back to sleep.

**Tech Stack:** Python, openWakeWord (ONNX), existing BaseHandler pipeline pattern

---

**Files:**
- Create: `src/speech_to_speech/WakeWord/__init__.py`
- Create: `src/speech_to_speech/WakeWord/wake_word_handler.py`
- Create: `src/speech_to_speech/arguments_classes/wake_word_arguments.py`
- Create: `tests/test_wake_word_handler.py`
- Modify: `src/speech_to_speech/arguments_classes/module_arguments.py:73-79`
- Modify: `src/speech_to_speech/s2s_pipeline.py`
- Modify: `src/speech_to_speech/pipeline/queue_types.py`
- Modify: `src/speech_to_speech/pipeline/handler_types.py`

### Task 1: Arguments dataclass for wake word config

**Files:**
- Create: `src/speech_to_speech/arguments_classes/wake_word_arguments.py`

- [ ] **Step 1: Create the file with four config fields**

```python
from dataclasses import dataclass, field


@dataclass
class WakeWordHandlerArguments:
    model_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to openWakeWord .tflite model file. When not set, wake word detection is disabled (current behavior)."
        },
    )
    threshold: float = field(
        default=0.5,
        metadata={
            "help": "Detection confidence threshold (0-1). Lower values increase sensitivity but may increase false positives."
        },
    )
    activation_timeout_s: float = field(
        default=30.0,
        metadata={
            "help": "Seconds of silence before the handler goes back to sleep after wake word activation."
        },
    )
    preroll_ms: int = field(
        default=1000,
        metadata={
            "help": "Milliseconds of audio to retain before the detected wake word and forward on activation."
        },
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/speech_to_speech/arguments_classes/wake_word_arguments.py
git commit -m "feat: add WakeWordHandlerArguments dataclass"
```

### Task 2: Queue type aliases

**Files:**
- Modify: `src/speech_to_speech/pipeline/handler_types.py:31`
- Modify: `src/speech_to_speech/pipeline/queue_types.py:25`

- [ ] **Step 1: Add WakeWordIn alias to handler_types.py**

```python
# ── WakeWord stage ────────────────────────────────────────────────
# WakeWordHandler consumes VADIn (bytes | tuple[bytes, RuntimeConfig])
# and produces the same type — it's a gate, not a transformer.
WakeWordIn: TypeAlias = bytes | tuple[bytes, RuntimeConfig]
```

Insert this block after line 31 in `handler_types.py`. The import for tuple is already present via `from __future__ import annotations`.

- [ ] **Step 2: Add WakeWordOutItem alias to queue_types.py**

```python
# Audio flowing from WakeWordHandler into VAD.
WakeWordOutItem: TypeAlias = VADIn | PipelineControlMessage
```

Insert this block after line 25 in `queue_types.py`.

- [ ] **Step 3: Run existing tests to confirm no breakage**

Run: `python -m pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add src/speech_to_speech/pipeline/handler_types.py src/speech_to_speech/pipeline/queue_types.py
git commit -m "feat: add wake word queue type aliases"
```

### Task 3: WakeWordHandler implementation

**Files:**
- Create: `src/speech_to_speech/WakeWord/__init__.py` (empty)
- Create: `src/speech_to_speech/WakeWord/wake_word_handler.py`

- [ ] **Step 1: Write the WakeWordHandler class**

```python
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from queue import Queue
from threading import Event
from typing import TYPE_CHECKING

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import VADIn

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class WakeWordHandler(BaseHandler[VADIn, VADIn]):
    """Gates audio between input source and VAD.

    In sleeping state, audio is fed to openWakeWord but not forwarded.
    On wake word detection, buffered + live audio is forwarded to VAD.
    After activation_timeout_s of silence (no chunks arriving), goes back to sleep.
    """

    def setup(
        self,
        model_path: str,
        threshold: float = 0.5,
        activation_timeout_s: float = 30.0,
        preroll_ms: int = 1000,
    ) -> None:
        self._threshold = threshold
        self._activation_timeout_s = activation_timeout_s
        self._state = "sleeping"
        self._buffer: list[bytes] = []
        self._last_forward_time: float = 0.0
        sample_rate = 16000
        self._preroll_chunks = max(1, int((preroll_ms / 1000) * sample_rate / 512))

        try:
            from openwakeword.model import Model

            self._model = Model(wakeword_models=[model_path])
            logger.info("WakeWordHandler: loaded model from %s", model_path)
        except Exception as e:
            logger.error("WakeWordHandler: failed to load model %s: %s", model_path, e)
            raise

    def process(self, item: VADIn) -> Iterator[VADIn]:
        chunk: bytes
        rt_cfg: RuntimeConfig | None = None
        if isinstance(item, tuple):
            chunk, rt_cfg = item
        else:
            chunk = item

        if self._state == "sleeping":
            prediction = self._model.predict(chunk)
            if prediction and max(prediction.values()) >= self._threshold:
                self._state = "active"
                self._last_forward_time = time.monotonic()
                for b in self._buffer:
                    yield (b, rt_cfg) if rt_cfg else b
                yield item
                self._buffer.clear()
                logger.info("WakeWordHandler: wake word detected, activating")
            else:
                self._buffer.append(chunk)
                while len(self._buffer) > self._preroll_chunks:
                    self._buffer.pop(0)

        elif self._state == "active":
            now = time.monotonic()
            if now - self._last_forward_time > self._activation_timeout_s:
                self._state = "sleeping"
                self._buffer.clear()
                logger.info("WakeWordHandler: activation timeout expired, going to sleep")
                return
            self._last_forward_time = now
            yield item

    def on_session_end(self) -> None:
        self._state = "sleeping"
        self._buffer.clear()
```

- [ ] **Step 2: Commit**

```bash
git add src/speech_to_speech/WakeWord/
git commit -m "feat: add WakeWordHandler implementation"
```

### Task 4: Wire WakeWordHandler into the pipeline

**Files:**
- Modify: `src/speech_to_speech/s2s_pipeline.py`

Strategy: `_build_pipeline_handlers` creates an intermediate `Queue` locally when wake word is enabled. WakeWordHandler reads from `recv_audio_chunks_queue`, writes to the intermediate queue, and VAD reads from the intermediate queue instead of `recv_audio_chunks_queue`. WakeWordHandler is prepended to the handler list.

- [ ] **Step 1: Add import at top of s2s_pipeline.py**

```python
from speech_to_speech.arguments_classes.wake_word_arguments import WakeWordHandlerArguments
```

- [ ] **Step 2: Modify `_build_pipeline_handlers` to accept wake word kwargs and create intermediate queue**

Add parameter to the function signature:

```python
def _build_pipeline_handlers(
    *,
    ...
    wake_word_handler_kwargs: WakeWordHandlerArguments | None = None,
) -> list[Any]:
```

At the top of the function body, after `from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor`, add:

```python
    if wake_word_handler_kwargs is not None and wake_word_handler_kwargs.model_path is not None:
        from speech_to_speech.WakeWord.wake_word_handler import WakeWordHandler

        ww_out_queue: Queue = Queue()
        wake_word = WakeWordHandler(
            stop_event,
            queue_in=recv_audio_chunks_queue,
            queue_out=ww_out_queue,
            setup_kwargs=vars(wake_word_handler_kwargs),
        )
        vad_in_queue: Queue = ww_out_queue
    else:
        wake_word = None
        vad_in_queue = recv_audio_chunks_queue

    vad = VADHandler(
        stop_event,
        queue_in=vad_in_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )
```

At the return, prepend the wake word handler:

```python
    handlers = [vad, stt, transcription_notifier, lm, lm_processor, tts]
    if wake_word is not None:
        handlers.insert(0, wake_word)
    return handlers
```

- [ ] **Step 3: Thread `wake_word_handler_kwargs` through `_build_realtime_pipeline_unit` and `build_pipeline`**

Add parameter to `_build_realtime_pipeline_unit`:

```python
def _build_realtime_pipeline_unit(
    *,
    ...
    wake_word_handler_kwargs: WakeWordHandlerArguments | None = None,
) -> "PipelineUnit":
```

Pass it to `_build_pipeline_handlers`:

```python
    handlers = _build_pipeline_handlers(
        ...
        wake_word_handler_kwargs=wake_word_handler_kwargs,
    )
```

Add parameter to `build_pipeline`:

```python
def build_pipeline(
    ...
    wake_word_handler_kwargs: WakeWordHandlerArguments | None = None,
    ...
```

In the realtime mode section, pass it:

```python
    pool = [
        _build_realtime_pipeline_unit(
            ...
            wake_word_handler_kwargs=wake_word_handler_kwargs,
        )
        for i in range(pool_size)
    ]
```

For local/socket/websocket modes, pass to `_build_pipeline_handlers` if it's called in those paths.

- [ ] **Step 4: Wire `wake_word_handler_kwargs` in `parse_arguments` and `main`**

In `parse_arguments`, add to the return:

```python
    return ParsedArguments(
        ...
        wake_word_handler_kwargs=WakeWordHandlerArguments(
            model_path=by_type[ModuleArguments].wake_word_model,
            threshold=by_type[ModuleArguments].wake_word_threshold,
            activation_timeout_s=by_type[ModuleArguments].wake_word_activation_timeout_s,
            preroll_ms=by_type[ModuleArguments].wake_word_preroll_ms,
        ),
    )
```

In `main`, pass `parsed_args.wake_word_handler_kwargs` to `build_pipeline` calls.

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech/s2s_pipeline.py
git commit -m "feat: wire WakeWordHandler into pipeline"
```

### Task 5: ModuleArguments and CLI

**Files:**
- Modify: `src/speech_to_speech/arguments_classes/module_arguments.py:73-79`
- Modify: `src/speech_to_speech/s2s_pipeline.py` (parse_arguments)

- [ ] **Step 1: Add wake_word fields to ModuleArguments**

Append to `module_arguments.py`:

```python
    wake_word_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to openWakeWord .tflite model file for wake word detection. "
            "When set, the system only responds after hearing the wake word. "
            "When not set (default), the system responds to any speech (current behavior)."
        },
    )
    wake_word_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Detection confidence threshold (0-1) for wake word activation. Default is 0.5."
        },
    )
    wake_word_activation_timeout_s: float = field(
        default=30.0,
        metadata={
            "help": "Seconds of silence before the wake word handler goes back to sleep. Default is 30."
        },
    )
    wake_word_preroll_ms: int = field(
        default=1000,
        metadata={
            "help": "Milliseconds of audio to retain before the detected wake word and forward on activation. Default is 1000."
        },
    )
```

- [ ] **Step 2: Add `WakeWordHandlerArguments` to `ParsedArguments` and `parse_arguments`**

Add field to `ParsedArguments`:

```python
    wake_word_handler_kwargs: WakeWordHandlerArguments
```

Add to `parse_arguments` return:

```python
        wake_word_handler_kwargs=WakeWordHandlerArguments(
            model_path=by_type[ModuleArguments].wake_word_model,
            threshold=by_type[ModuleArguments].wake_word_threshold,
            activation_timeout_s=by_type[ModuleArguments].wake_word_activation_timeout_s,
            preroll_ms=by_type[ModuleArguments].wake_word_preroll_ms,
        ),
```

And add import: `from speech_to_speech.arguments_classes.wake_word_arguments import WakeWordHandlerArguments`

- [ ] **Step 3: Run existing tests to verify no breakage**

Run: `python -m pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add src/speech_to_speech/arguments_classes/module_arguments.py src/speech_to_speech/arguments_classes/wake_word_arguments.py src/speech_to_speech/s2s_pipeline.py
git commit -m "feat: add wake word CLI arguments and wire through pipeline"
```

### Task 6: Unit tests for WakeWordHandler

**Files:**
- Create: `tests/test_wake_word_handler.py`

- [ ] **Step 1: Write basic handler construction test**

```python
from __future__ import annotations

from queue import Queue
from threading import Event

import pytest

from speech_to_speech.WakeWord.wake_word_handler import WakeWordHandler


def test_handler_requires_model_path():
    stop_event = Event()
    q_in: Queue = Queue()
    q_out: Queue = Queue()
    with pytest.raises(Exception):
        WakeWordHandler(stop_event, queue_in=q_in, queue_out=q_out, setup_args=(), setup_kwargs={})
```

- [ ] **Step 2: Write state machine tests**

```python
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
```

- [ ] **Step 3: Run tests to verify structure**

Run: `python -m pytest tests/test_wake_word_handler.py -v`
Expected: at least 2 tests pass

- [ ] **Step 4: Write state machine behavior tests**

```python
def test_sleeping_does_not_forward_bytes():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.5
    handler._activation_timeout_s = 30.0
    handler._last_forward_time = 0.0

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
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/test_wake_word_handler.py -v`
Expected: 5 tests pass

- [ ] **Step 7: Commit**

```bash
git add tests/test_wake_word_handler.py
git commit -m "test: add WakeWordHandler unit tests"
```

### Task 7: Integration test — pipeline with mocked WakeWordHandler

**Files:**
- Modify: `tests/test_wake_word_handler.py`

- [ ] **Step 1: Write a pipeline integration test**

```python
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
    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert results == []
    assert q_out.qsize() == 0
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/test_wake_word_handler.py -v`
Expected: all 6 tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_wake_word_handler.py
git commit -m "test: add WakeWordHandler integration test"
```
