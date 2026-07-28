# Audio Chimes Design

**Goal:** Play a WAV chime when the wake word is detected (signaling the user can speak) and when a server-side tool/search returns (signaling the system is about to speak the answer).

**Architecture:** Pre-load WAV files to 16 kHz int16 PCM bytes at startup. Inject raw PCM bytes directly into the shared `send_audio_chunks_queue` (same queue TTS output goes to) from the two trigger points — WakeWordHandler and the LM server-side tool loop. The existing send loop handles them identically to TTS audio.

**Chime flow (wake word):**
```
User: "computer"
  → WakeWordHandler detects, transitions to active
  → pushes chime bytes to send_audio_chunks_queue
  → forwards buffered audio to VAD (normal pipeline)
```

**Chime flow (search complete):**
```
LM calls search tool
  → yields "Searching" text (TTS synthesizes it)
  → tool returns with result
  → pushes chime bytes to send_audio_chunks_queue
  → LM generates response from tool result → TTS → output queue
```

**Timing:** The chime is in-band with TTS output; it plays after "Searching" TTS and before the answer TTS, making a clear earcon sequence: "Searching… ✨ … answer."

## Components

- **`ChimeLoader`** — loads WAV files, validates format, resamples to 16 kHz mono int16, caches as `bytes`. Two pre-load slots: `wake_chime` and `search_chime` (both optional).
- **`WakeWordHandler`** — new `wake_chime_bytes` and `chime_output_queue` setup params. On detection, puts chime bytes into the output queue.
- **`_generate()` loop** — new `search_chime_bytes` and `chime_output_queue` params. After each server-side tool execution, puts chime bytes into the output queue.
- **CLI** — `--wake_chime` and `--search_chime` under the wake word argument group, resolved in `WakeWordHandlerArguments`.

## Edge Cases

- **File not found / invalid WAV** — `ChimeLoader` logs a warning and treats the chime as absent (no crash).
- **No chime configured** — `None` bytes passed through; the trigger points skip the `put()` call.
- **Chime longer than anticipated** — Raw PCM is ~8 kB per 100 ms at 16 kHz. A 1-second chime (~160 kB) is fine; the send loop batches audio normally.
- **Race with TTS** — FIFO queue guarantees chime arrives before any subsequently synthesized output.

## Files

- Create: `src/speech_to_speech/chime_loader.py`
- Modify: `src/speech_to_speech/WakeWord/wake_word_handler.py`
- Modify: `src/speech_to_speech/LLM/base_openai_compatible_language_model.py`
- Modify: `src/speech_to_speech/arguments_classes/wake_word_arguments.py`
- Modify: `src/speech_to_speech/s2s_pipeline.py`
- Modify: `tests/test_wake_word_handler.py`
