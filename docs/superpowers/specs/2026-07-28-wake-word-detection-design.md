# Wake Word Detection Design

> **For agentic workers:** This is the design spec. After user approval, use `superpowers:writing-plans` to create the implementation plan.

**Goal:** Add optional wake-word ("computer") detection to the speech-to-speech pipeline so the system only responds after hearing its wake word, with a cooldown timeout before it goes back to sleep.

**Architecture:** Insert a `WakeWordHandler` between the audio source and VAD. It runs openWakeWord on incoming audio frames. When the wake word is detected, it starts forwarding audio to VAD. After a configurable period of silence, it goes back to sleep.

**Tech Stack:** Python, openWakeWord (ONNX), Silero VAD (optional noise gate)

---

## Pipeline Changes

Current: `Audio Source → VAD → STT → LLM → TTS`

With wake word: `Audio Source → WakeWordHandler → VAD → STT → LLM → TTS`

The `WakeWordHandler` is a new `BaseHandler` subclass. When no `--wake_word_model` is configured, the handler is omitted entirely (no behavioral change, no new dependency).

## WakeWordHandler States

### Sleeping
- Incoming audio frames are fed into openWakeWord for inference
- Audio is **not** forwarded to VAD
- On wake word detection (score >= threshold): transition to Active, emit a speech_started-like signal

### Active
- Buffered audio (pre-roll) is flushed to VAD
- Live audio is forwarded to VAD
- A `activation_timeout_s` timer runs: each new VAD speech segment resets it
- On timeout expiry: transition back to Sleeping

## Configuration (ModuleArguments / CLI)

```
--wake_word_model     Path to .tflite model file (default: None = disabled)
--wake_word_threshold Detection confidence (default: 0.5)
--activation_timeout_s How long to stay active after last speech (default: 30)
--wake_word_preroll_ms Audio buffer before detected wake word (default: 1000)
```

When `--wake_word_model` is not set, `WakeWordHandler` is not instantiated.

## Audio Flow

### Sleeping → Active transition
1. openWakeWord processes each 80ms frame
2. Score crosses threshold → transition to Active
3. A circular buffer (configurable `wake_word_preroll_ms`) holds the last N ms of audio
4. The buffered audio is flushed to VAD, followed by live audio

### Active → Sleeping transition
1. VAD detects speech_end
2. `activation_timeout_s` timer starts
3. If VAD detects speech_start before timeout: timer resets
4. If timer expires: transition to Sleeping

## Dependencies

- `openwakeword` — optional, only imported when `--wake_word_model` is set
- Pre-trained model: a placeholder (e.g., "alexa" from openWakeWord bundled models) for development; user trains a custom "computer" model later

## Files to Create/Modify

### New files
- `src/speech_to_speech/WakeWord/wake_word_handler.py` — handler implementation
- `src/speech_to_speech/WakeWord/__init__.py` — package init
- `src/speech_to_speech/arguments_classes/wake_word_arguments.py` — CLI argument dataclass
- `tests/test_wake_word_handler.py` — unit tests

### Modified files
- `src/speech_to_speech/arguments_classes/module_arguments.py` — add wake word config fields
- `src/speech_to_speech/s2s_pipeline.py` — wire handler into pipeline build + pass args
- `src/speech_to_speech/baseHandler.py` or relevant pipeline construction — add wake word to handler chain

## Error Handling

- openWakeWord model file not found → log error and fall back to always-on (or fail at startup)
- openWakeWord inference failure → log warning, forward audio to VAD (fail open)
- Invalid threshold values → clamp to [0, 1] range
- Audio format mismatch → resample to 16kHz 16-bit PCM as required by openWakeWord

## Testing Strategy

### Unit tests (WakeWordHandler in isolation)
- Sleeping state: audio not forwarded to output queue
- Detection: transition to Active, buffered audio flushed
- Cooldown: timeout after silence, back to Sleeping
- Configurable threshold, timeout, pre-roll
- No model configured = handler not created (via `get_wake_word_handler()`)

### Integration
- Full pipeline with mocked wake word model: sleep → detect → forward → cooldown → sleep
- Wake word + server-side web search interaction
