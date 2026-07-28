# Audio Chimes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play configurable WAV chimes on wake word detection (user can speak) and after server-side tool execution (answer incoming).

**Architecture:** `ChimeLoader` loads WAV → 16 kHz PCM bytes at startup. Both `WakeWordHandler` and the LM `_generate()` loop push chime bytes directly into `send_audio_chunks_queue` (the same queue TTS output uses). When no chime path is configured, bytes are `None` and the push is skipped.

**Tech Stack:** Python `wave` (stdlib), `numpy`, existing queue architecture.

---

### Task 1: ChimeLoader utility

**Files:**
- Create: `src/speech_to_speech/chime_loader.py`
- Test: `tests/test_chime_loader.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for ChimeLoader — WAV loading, resampling, and error handling."""
from __future__ import annotations

import numpy as np
import pytest
import wave

from speech_to_speech.chime_loader import ChimeLoader


def _make_wav(path: str, samples: np.ndarray, sr: int) -> None:
    """Helper: write a mono 16-bit WAV file."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())


def test_chime_loader_returns_none_for_no_path(tmp_path):
    loader = ChimeLoader(wake_chime_path=None, search_chime_path=None)
    assert loader.wake_chime is None
    assert loader.search_chime is None


def test_chime_loader_loads_16khz_wav(tmp_path):
    p = str(tmp_path / "wake.wav")
    samples = np.arange(16000, dtype=np.int16)  # 1 second at 16 kHz
    _make_wav(p, samples, 16000)
    loader = ChimeLoader(wake_chime_path=p, search_chime_path=None)
    assert isinstance(loader.wake_chime, bytes)
    assert len(loader.wake_chime) == 16000 * 2


def test_chime_loader_resamples_48khz_to_16khz(tmp_path):
    p = str(tmp_path / "wake.wav")
    samples = np.arange(48000, dtype=np.int16)  # 1 second at 48 kHz
    _make_wav(p, samples, 48000)
    loader = ChimeLoader(wake_chime_path=p, search_chime_path=None)
    assert isinstance(loader.wake_chime, bytes)
    assert len(loader.wake_chime) == 16000 * 2  # resampled to 16 kHz


def test_chime_loader_missing_file_logs_warning(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    loader = ChimeLoader(wake_chime_path=str(tmp_path / "nonexistent.wav"), search_chime_path=None)
    assert loader.wake_chime is None
    assert "nonexistent.wav" in caplog.text


def test_chime_loader_invalid_wav_logs_warning(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    p = str(tmp_path / "bad.wav")
    with open(p, "wb") as f:
        f.write(b"not a wav file")
    loader = ChimeLoader(wake_chime_path=p, search_chime_path=None)
    assert loader.wake_chime is None
    assert "bad.wav" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chime_loader.py -v`
Expected: ModuleNotFoundError / ImportError for ChimeLoader

- [ ] **Step 3: Write ChimeLoader**

```python
"""Load WAV files and normalise to 16 kHz mono int16 PCM bytes."""
from __future__ import annotations

import logging
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PIPELINE_RATE = 16000


def _load_wav(path: str) -> Optional[bytes]:
    """Read a WAV file, normalise to 16 kHz mono int16 PCM, return bytes or None on error."""
    try:
        with wave.open(path, "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except Exception as e:
        logger.warning("ChimeLoader: failed to read WAV %s: %s", path, e)
        return None

    if sampwidth != 2:
        logger.warning("ChimeLoader: unsupported sample width %d for %s (must be 16-bit)", sampwidth, path)
        return None

    audio = np.frombuffer(frames, dtype=np.int16)

    # Downmix multi-channel to mono
    if nchannels > 1:
        audio = audio.reshape(-1, nchannels).mean(axis=1).astype(np.int16)

    # Resample if needed
    if framerate != PIPELINE_RATE:
        duration_s = len(audio) / framerate
        target_len = int(duration_s * PIPELINE_RATE)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio.astype(np.float32),
        ).astype(np.int16)

    return audio.tobytes()


class ChimeLoader:
    """Pre-load chime WAVs to 16 kHz int16 PCM bytes.

    ``wake_chime`` / ``search_chime`` properties return ``bytes | None``.
    """

    def __init__(self, wake_chime_path: str | None = None, search_chime_path: str | None = None) -> None:
        self._wake_chime: bytes | None = None
        self._search_chime: bytes | None = None
        if wake_chime_path:
            self._wake_chime = _load_wav(wake_chime_path)
        if search_chime_path:
            self._search_chime = _load_wav(search_chime_path)

    @property
    def wake_chime(self) -> bytes | None:
        return self._wake_chime

    @property
    def search_chime(self) -> bytes | None:
        return self._search_chime
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chime_loader.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech/chime_loader.py tests/test_chime_loader.py
git commit -m "feat: add ChimeLoader utility for WAV loading and resampling"
```

---

### Task 2: CLI arguments for chime paths

**Files:**
- Modify: `src/speech_to_speech/arguments_classes/wake_word_arguments.py`
- Update: `tests/test_cli_defaults.py` (expected default count)

- [ ] **Step 1: Add wake_chime and search_chime fields**

```python
@dataclass
class WakeWordHandlerArguments:
    # ... existing fields ...
    wake_chime: str | None = field(
        default=None,
        metadata={"help": "Path to WAV file to play when wake word is detected (signals user can speak)."},
    )
    search_chime: str | None = field(
        default=None,
        metadata={"help": "Path to WAV file to play after a server-side tool/search returns (signals answer incoming)."},
    )
```

- [ ] **Step 2: Update CLI defaults test**

Run: `uv run pytest tests/test_cli_defaults.py -v`
Find the expected WakeWordHandlerArguments field count, increment by 2 if it exists.

- [ ] **Step 3: Commit**

```bash
git add src/speech_to_speech/arguments_classes/wake_word_arguments.py
git commit -m "feat: add --wake_chime and --search_chime CLI arguments"
```

---

### Task 3: Wire chimes into pipeline building

**Files:**
- Modify: `src/speech_to_speech/s2s_pipeline.py`

- [ ] **Step 1: Wire ChimeLoader and pass to handlers**

In `_build_pipeline_handlers()` (line ~388), after constructing `wake_word_handler_kwargs`, instantiate the ChimeLoader and pass chime bytes + queue ref to both handlers:

```python
from speech_to_speech.chime_loader import ChimeLoader

# After the wake_word_handler_kwargs block (~line 397):
wake_word_chime: bytes | None = None
search_chime: bytes | None = None
if wake_word_handler_kwargs is not None:
    ww_path = getattr(wake_word_handler_kwargs, "wake_chime", None)
    sc_path = getattr(wake_word_handler_kwargs, "search_chime", None)
    if ww_path or sc_path:
        chime_loader = ChimeLoader(wake_chime_path=ww_path, search_chime_path=sc_path)
        wake_word_chime = chime_loader.wake_chime
        search_chime = chime_loader.search_chime
```

Then pass `send_audio_chunks_queue` and the chime bytes through `setup_kwargs`:

For WakeWordHandler (line ~405):
```python
ww_setup = vars(wake_word_handler_kwargs)
if wake_word_chime is not None:
    ww_setup["chime_output_queue"] = send_audio_chunks_queue
    ww_setup["wake_chime_bytes"] = wake_word_chime
wake_word = WakeWordHandler(stop_event, queue_in=recv_audio_chunks_queue, queue_out=ww_out_queue, setup_kwargs=ww_setup)
```

For LM handler: `get_llm_handler()` returns a handler that accepts `**_kwargs` in setup(). We need to add chime params to the handler after construction or pass them through kwargs. Since `get_llm_handler` passes `vars(responses_api_language_model_handler_kwargs)`, we extend those before the call:

```python
if search_chime is not None:
    responses_api_language_model_handler_kwargs.chime_output_queue = send_audio_chunks_queue
    responses_api_language_model_handler_kwargs.search_chime_bytes = search_chime
```

But those are dataclasses... we can just set attributes. Or better, use `vars()` on the handler kwargs directly if they're mutable dicts.

Actually, the pattern used already in `_build_realtime_pipeline_unit` is:
```python
vars(kw)["cancel_scope"] = cancel_scope
```

So we can do the same for chime params on the kw dataclass before passing to `get_llm_handler`.

Actually wait - the chime params need to be added to the handler's setup_kwargs dict, not the dataclass. Let me look at how `get_llm_handler` works again...

In `get_llm_handler()`:
```python
ResponsesApiModelHandler(
    ...
    setup_kwargs=vars(responses_api_language_model_handler_kwargs),
)
```

So `setup_kwargs` is the dict from `vars()`. If I add extra entries to `vars(dataclass_instance)`, they'll be included. But `vars()` returns a dict that is NOT the dataclass fields - mutating it mutates the instance's `__dict__`. For a frozen dataclass this might not work, but for a regular one it does.

Actually, the simpler approach: in `_build_pipeline_handlers`, after constructing chime loader, pass chime queue/bytes as additional setup_kwargs by modifying the vars dicts.

For the LM handler, I'll modify `_build_realtime_pipeline_unit()` (where the per-unit copies live) since that's where `vars(kw)["cancel_scope"]` etc. are already set. The chime params should be set there too.

Actually wait, `_build_pipeline_handlers` is called from both `_build_realtime_pipeline_unit` and the normal `build_pipeline` path. Let me handle both paths.

Let me just modify `_build_pipeline_handlers` directly to handle the chime wiring. That's simpler - one place.

Actually - looking at the code again, the `_build_pipeline_handlers` function takes individual kwarg dataclasses as parameters. The function creates the handlers by passing `vars(dataclass)` as `setup_kwargs`. I can modify the vars dict before passing.

Let me just modify `_build_pipeline_handlers` to add chime params after creating the loader.

- [ ] **Step 2: Verify tests still pass**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/speech_to_speech/s2s_pipeline.py
git commit -m "feat: wire ChimeLoader into pipeline handlers"
```

---

### Task 4: WakeWordHandler chime injection

**Files:**
- Modify: `src/speech_to_speech/WakeWord/wake_word_handler.py`
- Modify: `tests/test_wake_word_handler.py`

- [ ] **Step 1: Add wake_chime_bytes and chime_output_queue to WakeWordHandler.setup()**

```python
def setup(
    self,
    model_path: str,
    threshold: float = 0.5,
    activation_timeout_s: float = 30.0,
    preroll_ms: int = 1000,
    wake_chime_bytes: bytes | None = None,
    chime_output_queue: Queue | None = None,
) -> None:
    self._wake_chime_bytes = wake_chime_bytes
    self._chime_output_queue = chime_output_queue
    # ... rest unchanged
```

- [ ] **Step 2: Push chime on wake word detection**

In the `process()` method, after the wake word is detected and before yielding buffered audio:

```python
if self._state == "sleeping":
    audio_array = np.frombuffer(chunk, dtype=np.int16)
    prediction = self._model.predict(audio_array)
    if prediction and max(prediction.values()) >= self._threshold:
        self._state = "active"
        self._last_forward_time = time.monotonic()
        # Push wake chime to output queue (best-effort, non-blocking)
        if self._wake_chime_bytes is not None and self._chime_output_queue is not None:
            try:
                self._chime_output_queue.put_nowait(self._wake_chime_bytes)
            except Exception:
                pass
        for b in self._buffer:
            yield (b, rt_cfg) if rt_cfg else b
        yield item
        self._buffer.clear()
```

- [ ] **Step 3: Update existing tests to pass None for new params**

The tests use `WakeWordHandler.__new__` and set attributes directly, so no changes needed unless a test calls `setup()`. The mock tests should work as-is since `WakeWordHandler.__new__` creates a bare object and we set `._state` etc. manually. Let the existing tests pass as-is.

- [ ] **Step 4: Add a test for chime injection**

```python
def test_wake_word_handler_plays_chime_on_detection():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.0  # always trigger
    handler._activation_timeout_s = 30.0
    handler._last_forward_time = 0.0
    handler._model = _mock_model(score=0.9)
    chime_bytes = b"\x00\x01\x02\x03"
    chime_queue: Queue = Queue()
    handler._wake_chime_bytes = chime_bytes
    handler._chime_output_queue = chime_queue

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert len(results) == 2  # buffered (empty) + live chunk
    assert chime_queue.qsize() == 1
    assert chime_queue.get_nowait() == chime_bytes


def test_wake_word_handler_no_chime_when_not_configured():
    handler = WakeWordHandler.__new__(WakeWordHandler)
    handler._state = "sleeping"
    handler._buffer = []
    handler._preroll_chunks = 5
    handler._threshold = 0.0
    handler._activation_timeout_s = 30.0
    handler._last_forward_time = 0.0
    handler._model = _mock_model(score=0.9)
    handler._wake_chime_bytes = None
    handler._chime_output_queue = None

    chunk = b"\x00" * 1024
    results = list(handler.process(chunk))
    assert len(results) == 2
    # No chime queue, no error
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/test_wake_word_handler.py tests/test_chime_loader.py -v`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add src/speech_to_speech/WakeWord/wake_word_handler.py tests/test_wake_word_handler.py
git commit -m "feat: inject wake chime on wake word detection"
```

---

### Task 5: LM handler search chime injection

**Files:**
- Modify: `src/speech_to_speech/LLM/base_openai_compatible_language_model.py`

- [ ] **Step 1: Accept search_chime_bytes and chime_output_queue in setup()**

The setup() signature already has `**_kwargs: Any` which catches extra kwargs. We store them as instance attributes:

```python
def setup(
    self,
    # ... existing params ...
    **kwargs: Any,
) -> None:
    # ... existing init ...
    self._search_chime_bytes: bytes | None = kwargs.pop("search_chime_bytes", None)
    self._chime_output_queue: Queue | None = kwargs.pop("chime_output_queue", None)
```

- [ ] **Step 2: Push search chime after server-side tool execution**

In `_generate()`, after the tool output is appended to history (~line 545):

```python
output = handler(**args)
tool_output = RealtimeConversationItemFunctionCallOutput(
    id=_generate_id("msg"),
    call_id=tool.call_id,
    output=output,
    type="function_call_output",
)
original_chat.append_tool_output(tool.call_id, tool_output)

# Push search chime to output queue
if self._search_chime_bytes is not None and self._chime_output_queue is not None:
    try:
        self._chime_output_queue.put_nowait(self._search_chime_bytes)
    except Exception:
        pass
```

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest tests/test_chat_completions_backend.py tests/test_responses_api_language_model.py tests/test_lm_output_processor.py -v`
Expected: All passed

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/speech_to_speech/LLM/base_openai_compatible_language_model.py
git commit -m "feat: inject search chime after server-side tool execution"
```
