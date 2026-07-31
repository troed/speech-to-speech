# *"Computer!"* - a low latency tool capable voice assistant

If you've ever watched Star Trek: The Next Generation and felt that you wanted your own *Computer!* - here you are. This started out as a fork of the [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) project where every change and addition that has been made has had that goal in mind.

Notable functionality:

**1. Wake word detection.** The system only responds initially after hearing a configured wake word (e.g. "computer"). Uses [openWakeWord](https://github.com/dscripka/openWakeWord) — train a custom ONNX model at [openwakeword.com/train](https://openwakeword.com/train). During the conversation that follows no wake word is needed until after a configurable cooldown period.

**2. Audio chimes.** Plays a WAV chime on wake word detection (signals the user can speak) and when a server-side tool/search returns (signals the answer is incoming).

**3. Barge-in (interrupt).** Speak over the assistant mid-response to interrupt TTS output. The system cancels in-flight generation, clears pending audio, plays a wake chime, and processes your speech as new input without requiring the wake word.

**4. Dedicated WebSocket client(s).** A standalone `speech-to-speech-client` CLI tool handles wake word detection, microphone input, and speaker output. One server can respond to multiple clients.

**5. Can use multimodal LLM backend for STT.** If the backing LLM is multimodal, like Gemma4 12B, no dedicated Speech-To-Text (STT) component is needed. This saves VRAM and helps allowing the full stack to run on regular consumer GPUs.

Caveat: For obvious reasons this project does not include the specific chimes and voice cloning data you might want to use.

## Example configurations

Three stacked configurations using different amounts of GPU VRAM. Every option runs VAD (Silero VAD v5, ~6 MB), so it is not listed separately. The LLM (Gemma4 12B) runs as a separate llama-server process; all STT and TTS models are shared across pipeline units.

### Option 1: faster-whisper + Qwen3-TTS

| Component | VRAM |
|---|---|
| faster-whisper tiny.en | ~150 MB |
| Qwen3-TTS Q8_0 | ~3.2 GB |
| **Pipeline total** | **~3.4 GB** |

```bash
speech-to-speech \
    --stt faster-whisper --faster_whisper_stt_model_name tiny.en \
    --tts qwen3 --qwen3_tts_backend ggml --qwen3_tts_quant Q8_0 \
    --mode realtime
```

### Option 2: Parakeet TDT + Qwen3-TTS

| Component | VRAM |
|---|---|
| Parakeet TDT 0.6B | ~2 GB |
| Qwen3-TTS Q8_0 | ~3.2 GB |
| **Pipeline total** | **~5.2 GB** |

```bash
speech-to-speech \
    --stt parakeet-tdt \
    --tts qwen3 --qwen3_tts_backend ggml --qwen3_tts_quant Q8_0 \
    --mode realtime
```

### Option 3: Native LLM audio (no STT) + Qwen3-TTS FP16

Drops the STT model entirely. Audio is transcribed by the multimodal LLM itself. Frees enough VRAM to run Qwen3-TTS at full FP16 precision.

| Component | VRAM |
|---|---|
| Qwen3-TTS FP16 | ~3.6 GB |
| **Pipeline total** | **~3.6 GB** |

```bash
speech-to-speech \
    --stt native-llm --llm_backend chat-completions \
    --tts qwen3 --qwen3_tts_backend ggml \
    --mode realtime
```

### Reference server configuration

The file `server.sh` in the repository root is the author's daily-driver configuration:

- **LLM**: Gemma4 12B QAT served by llama-server on another machine (`--llm_backend responses-api`, `--responses_api_base_url http://192.168.0.2:11434`).
- **STT**: faster-whisper tiny.en (`--stt faster-whisper`, `--faster_whisper_stt_model_name tiny.en`).
- **TTS**: Qwen3-TTS with voice cloning to a Star Trek TNG computer voice (`--tts qwen3`, `--qwen3_tts_quant Q8_0`, `--qwen3_tts_ref_audio computer.wav`).
- **System prompt**: Star Trek TNG computer persona with tool-use instructions (`--init_chat_prompt`).
- **MCP tools**: Home Assistant and web search via two MCP servers (`--mcp-config mcp.json`).
- **Two concurrent clients**: (`--num_pipelines 2`).
- **Wake word**: Wake word model handled on the client side; server plays chimes on wake (`--wake_word_wake_chime here.wav`) and after tool results (`--wake_word_search_chime ready.wav`).

Adjust `--responses_api_base_url` to point at your own llama-server, swap in your own `--qwen3_tts_ref_audio` and `--qwen3_tts_instruct` for voice cloning, and replace `--init_chat_prompt` with your desired persona.

### Client configuration

A companion `speech-to-speech-client` CLI handles microphone, speaker, and optional local wake word:

```bash
speech-to-speech-client \
    --host 192.168.0.2 --port 8766 \
    --wake-word-model computer.onnx \
    --wake-inactivity-timeout 10
```

Connect multiple clients at the same time: the server pool (`--num_pipelines`) determines how many can talk simultaneously. Extra clients are rejected when the pool is full.

### Model Setup

The pipeline uses several models (STT, VAD, TTS). These are **never auto-downloaded** — all network access is blocked at startup. There are two ways to get them:

**One-time download (recommended):**

```bash
uv run python scripts/download_models.py
```

Downloads all default models (NLTK data, Silero VAD, Parakeet TDT, Qwen3-TTS) into local caches. After this completes, the pipeline runs fully offline.

## Realtime API

Realtime mode streams audio over a WebSocket using the OpenAI Realtime protocol, with live transcription and low-latency turn-taking. The server exposes `/v1/realtime`, and any OpenAI Realtime-compatible client can connect. A dedicated `speech-to-speech-client` CLI is included.

See [src/speech_to_speech/api/openai_realtime/README.md](./src/speech_to_speech/api/openai_realtime/README.md) for the supported event set, architecture, and design details.

## License

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

## Citations (from the upstream Speech-to-Speech project)

If you use this pipeline, please also cite the component models you run. The defaults are:

### Gemma4

```bibtex
@misc{gemma4-2026,
  author = {Google DeepMind},
  title = {Gemma 4},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/google/gemma-4-12B-it}}
}
```

### Silero VAD

```bibtex
@misc{SileroVAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  email = {hello@silero.ai}
}
```

### Parakeet TDT

```bibtex
@misc{parakeet-tdt,
  author = {NVIDIA},
  title = {Parakeet TDT 0.6B v3},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3}}
}
```

### Qwen3-TTS

```bibtex
@misc{qwen3-tts,
  author = {Qwen Team},
  title = {Qwen3-TTS},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}}
}
```

Citations for optional backends such as Kokoro, Pocket TTS, ChatTTS, Whisper variants, Paraformer, and MMS live in the respective [component READMEs](./src/speech_to_speech).
