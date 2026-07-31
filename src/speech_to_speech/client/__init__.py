"""CLI client for speech-to-speech OpenAI Realtime API mode.

Usage:
  speech-to-speech-client --host 192.168.0.2 --port 8766
  speech-to-speech-client --wake-word-model computer.onnx
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
import wave
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

PIPELINE_RATE = 16000


def _encode_input_audio(chunk: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(chunk).decode("ascii"),
    }


def _decode_output_audio(event: dict[str, Any]) -> bytes | None:
    if event.get("type") != "response.output_audio.delta":
        return None
    b64 = event.get("delta")
    if not b64:
        return None
    return base64.b64decode(b64)


def _parse_realtime_text_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    event_type = event.get("type", "")

    if event_type == "conversation.item.input_audio_transcription.delta":
        return {"kind": "partial_transcription", "delta": event.get("delta", "")}

    if event_type == "conversation.item.input_audio_transcription.completed":
        return {"kind": "transcription_completed", "transcript": event.get("transcript", "")}

    if event_type == "input_audio_buffer.speech_started":
        return {"kind": "speech_started"}

    if event_type == "response.output_audio_transcript.done":
        return {"kind": "assistant_text", "text": event.get("transcript", "")}

    if event_type == "response.done":
        resp = event.get("response", {})
        status = resp.get("status", "")
        if status == "failed":
            error_msg = ((resp.get("status_details") or {}).get("error") or {}).get("message", "Unknown error")
            return {"kind": "response_failed", "error": error_msg}
        if status == "completed":
            result: dict[str, Any] = {"kind": "response_done"}
            usage = resp.get("usage")
            if usage:
                result["input_tokens"] = usage.get("input_tokens", 0)
                result["output_tokens"] = usage.get("output_tokens", 0)
            return result
        return {"kind": "response_done"}

    return None


def _build_session_update(
    voice: str | None = None,
    instructions: str | None = None,
) -> dict[str, Any]:
    session: dict[str, Any] = {
        "type": "realtime",
        "audio": {
            "input": {
                "turn_detection": {"type": "server_vad"},
            },
            "output": {},
        },
        "output_modalities": ["audio", "text"],
    }
    if voice is not None:
        session["voice"] = voice
    if instructions is not None:
        session["instructions"] = instructions
    return {"type": "session.update", "session": session}


def _load_wav(path: str) -> bytes | None:
    try:
        with wave.open(path, "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except Exception as e:
        logger.warning("Failed to read WAV %s: %s", path, e)
        return None

    if sampwidth != 2:
        logger.warning("Unsupported sample width %d for %s (must be 16-bit)", sampwidth, path)
        return None

    audio = np.frombuffer(frames, dtype=np.int16)

    if nchannels > 1:
        audio = audio.reshape(-1, nchannels).mean(axis=1).astype(np.int16)

    if framerate != PIPELINE_RATE:
        duration_s = len(audio) / framerate
        target_len = int(duration_s * PIPELINE_RATE)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio.astype(np.float32),
        ).astype(np.int16)

    return audio.tobytes()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class AudioStreamer:
    """Sounddevice-based mic input + speaker output.

    Mic audio is pushed to ``mic_queue``. Speaker audio is pulled from
    ``speaker_queue`` and played with output priority (half-duplex behavior
    identical to the server-side LocalAudioStreamer).
    """

    def __init__(
        self,
        mic_queue: Queue[bytes],
        speaker_queue: Queue[bytes],
        sample_rate: int = 16000,
        chunk_size: int = 512,
        input_device: int | None = None,
        output_device: int | None = None,
    ) -> None:
        self.mic_queue = mic_queue
        self.speaker_queue = speaker_queue
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_device = input_device
        self.output_device = output_device
        self._pcm_buffer = bytearray()
        self._dither = np.random.randint(-1, 2, size=(chunk_size, 1), dtype=np.int16)

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
        try:
            self.mic_queue.put_nowait(bytes(indata))
        except Exception:
            pass

        while not self.speaker_queue.empty():
            try:
                chunk = self.speaker_queue.get_nowait()
                self._pcm_buffer.extend(chunk)
            except Empty:
                break

        if self._pcm_buffer:
            needed = frames * 2
            play = bytes(self._pcm_buffer[:needed])
            self._pcm_buffer = self._pcm_buffer[needed:]
            outdata[:] = np.frombuffer(play + b"\x00" * (needed - len(play)), dtype=np.int16).reshape(-1, 1)
        else:
            outdata[:] = self._dither

    def run(self, stop_event: Event) -> None:
        with sd.Stream(
            samplerate=self.sample_rate,
            dtype="int16",
            channels=1,
            blocksize=self.chunk_size,
            callback=self._callback,
            device=(self.input_device, self.output_device),
        ):
            stop_event.wait()


class WakeWordDetector:
    """Local wake word detection via openWakeWord.

    Runs in a separate thread, feeding mic audio to the model and notifying
    the main loop when detected.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        preroll_ms: int = 1000,
        cooldown_s: float = 2.0,
        wake_chime_bytes: bytes | None = None,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.preroll_chunks = max(1, preroll_ms // 32)
        self.cooldown_s = cooldown_s
        self._wake_chime_bytes = wake_chime_bytes
        self._speaker_queue: Queue[bytes] | None = None
        self._model: Any = None
        self._buffer: list[bytes] = []
        self._last_detection: float = 0.0

    def start(
        self, mic_queue: Queue[bytes], wake_event: Event, speaker_queue: Queue[bytes] | None = None
    ) -> None:
        try:
            from openwakeword.model import Model
        except ImportError:
            raise ImportError(
                "openwakeword is required for wake word support. "
                "Install with: pip install openwakeword"
            ) from None

        self._model = Model(wakeword_model_paths=[self.model_path])
        self._speaker_queue = speaker_queue
        logger.info("Wake word model loaded from %s", self.model_path)

        silence_audio = np.zeros(512, dtype=np.int16)

        def _run() -> None:
            was_active = False
            while True:
                if wake_event.is_set():
                    was_active = True
                    time.sleep(0.1)
                    continue
                if was_active:
                    was_active = False
                    self._model.reset()
                    for _ in range(31):
                        self._model.predict(silence_audio)
                    while True:
                        try:
                            mic_queue.get_nowait()
                        except Empty:
                            break
                try:
                    chunk = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                now = time.monotonic()
                if now < self._last_detection + self.cooldown_s:
                    self._buffer.append(chunk)
                    while len(self._buffer) > self.preroll_chunks:
                        self._buffer.pop(0)
                    continue

                audio_array = np.frombuffer(chunk, dtype=np.int16)
                prediction = self._model.predict(audio_array)
                if prediction and max(prediction.values()) >= self.threshold:
                    if not wake_event.is_set():
                        wake_event.set()
                        self._last_detection = now
                        logger.info("Wake word detected, activating")
                        self._model.reset()
                        self._play_wake_chime()
                        while True:
                            try:
                                mic_queue.get_nowait()
                            except Empty:
                                break
                else:
                    self._buffer.append(chunk)
                    while len(self._buffer) > self.preroll_chunks:
                        self._buffer.pop(0)

        Thread(target=_run, daemon=True).start()

    def _play_wake_chime(self) -> None:
        if self._wake_chime_bytes is not None and self._speaker_queue is not None:
            try:
                self._speaker_queue.put_nowait(self._wake_chime_bytes)
                logger.info("Played wake chime (%d bytes)", len(self._wake_chime_bytes))
            except Exception:
                logger.warning("Failed to queue wake chime", exc_info=True)


async def websocket_client(
    host: str,
    port: int,
    mic_queue: Queue[bytes],
    speaker_queue: Queue[bytes],
    audio_streamer: AudioStreamer,
    stop_event: Event,
    wake_event: Event | None = None,
    wake_inactivity_timeout: float = 10.0,
) -> None:
    import websockets.exceptions
    from websockets.asyncio.client import connect

    url = f"ws://{host}:{port}/v1/realtime"
    live_user_width = 0
    response_active = False
    last_recv_audio: float = 0.0
    grace_deadline: float = 0.0

    def render_user(text: str, final: bool = False) -> None:
        nonlocal live_user_width
        line = f"USER: {text}"
        padded = line if len(line) >= live_user_width else line + " " * (live_user_width - len(line))
        if final:
            print(f"\r{padded}", flush=True)
            live_user_width = 0
        else:
            print(f"\r{padded}", end="", flush=True)
            live_user_width = len(line)

    def clear_live() -> None:
        nonlocal live_user_width
        if live_user_width > 0:
            print("\r" + " " * live_user_width + "\r", end="", flush=True)
            live_user_width = 0

    def _drain_queue(queue: Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break

    def _extend_grace() -> None:
        nonlocal grace_deadline
        if grace_deadline > 0:
            grace_deadline = time.monotonic() + wake_inactivity_timeout

    async def send_audio(ws: Any) -> None:
        nonlocal grace_deadline
        while not stop_event.is_set():
            if wake_event is not None:
                if not wake_event.is_set():
                    await asyncio.sleep(0.1)
                    continue

                if response_active and time.monotonic() - last_recv_audio > 1.5 and speaker_queue.empty():
                    if grace_deadline <= time.monotonic():
                        grace_deadline = time.monotonic() + wake_inactivity_timeout
                        logger.info("Response complete, %ss grace period", wake_inactivity_timeout)
                    await asyncio.sleep(0.1)
                    continue

                if not response_active and grace_deadline > 0 and time.monotonic() >= grace_deadline:
                    grace_deadline = 0
                    wake_event.clear()
                    _drain_queue(mic_queue)
                    logger.info("Grace period expired, waiting for wake word")
                    continue

            try:
                chunk = await asyncio.to_thread(mic_queue.get, True, 0.1)
            except Empty:
                continue

            if wake_event is None or wake_event.is_set():
                event = _encode_input_audio(chunk)
                await ws.send(json.dumps(event))

    async def receive_audio(ws: Any) -> None:
        nonlocal last_recv_audio, response_active, grace_deadline
        while not stop_event.is_set():
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("Server disconnected")
                stop_event.set()
                break

            if isinstance(message, bytes):
                speaker_queue.put_nowait(message)
                last_recv_audio = time.monotonic()
                continue

            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                continue

            audio = _decode_output_audio(event)
            if audio is not None:
                speaker_queue.put_nowait(audio)
                last_recv_audio = time.monotonic()
                if not response_active:
                    response_active = True
                if wake_event is not None and not wake_event.is_set():
                    wake_event.set()
                continue

            parsed = _parse_realtime_text_event(event)
            if parsed is None:
                continue

            kind = parsed["kind"]
            if kind == "partial_transcription":
                text = parsed.get("delta", "")
                if text:
                    render_user(text)
                _extend_grace()
            elif kind == "transcription_completed":
                text = parsed.get("transcript", "")
                if text:
                    render_user(text, final=True)
                _extend_grace()
            elif kind == "speech_started":
                _extend_grace()
            elif kind == "assistant_text":
                clear_live()
                print(f"ASSISTANT: {parsed.get('text', '')}", flush=True)
            elif kind == "response_failed":
                clear_live()
                print(f"ERROR: {parsed.get('error', 'Unknown error')}", flush=True)
            elif kind == "response_done":
                if response_active:
                    grace_deadline = time.monotonic() + wake_inactivity_timeout
                if "input_tokens" in parsed:
                    logger.debug(
                        "Tokens: %d in / %d out",
                        parsed.get("input_tokens", 0),
                        parsed.get("output_tokens", 0),
                    )
                response_active = False

    while not stop_event.is_set():
        try:
            logger.info("Connecting to %s ...", url)
            async with connect(url) as ws:
                logger.info("Connected")
                await ws.send(json.dumps(_build_session_update()))
                clear_live()
                print("Connected. Press Ctrl+C to stop.", flush=True)

                recv_task = asyncio.create_task(receive_audio(ws))
                send_task = None

                if wake_event is not None:
                    logger.info("Waiting for wake word...")
                    while not wake_event.is_set() and not stop_event.is_set():
                        await asyncio.sleep(0.1)
                    if stop_event.is_set():
                        recv_task.cancel()
                        try:
                            await recv_task
                        except asyncio.CancelledError:
                            pass
                        break

                send_task = asyncio.create_task(send_audio(ws))
                tasks = [send_task, recv_task]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except (ConnectionRefusedError, OSError, websockets.exceptions.InvalidURI) as exc:
            logger.error("Connection failed: %s", exc)
            if wake_event is None:
                logger.info("Retrying in 3 seconds...")
                await asyncio.sleep(3)
            else:
                logger.info("Waiting for wake word to reconnect...")
                wake_event.clear()
                while not wake_event.is_set() and not stop_event.is_set():
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech-to-speech CLI client (WebSocket mode)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8766, help="Server WebSocket port (realtime API)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--chunk-size", type=int, default=512, help="Audio chunk size (samples)")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=None, help="sounddevice output device index")
    parser.add_argument("--wake-word-model", default=None, help="Path to openWakeWord .tflite model")
    parser.add_argument("--wake-word-threshold", type=float, default=0.5, help="Wake word threshold (0-1)")
    parser.add_argument("--wake-chime", default=None, help="Path to WAV file played on wake word detection")
    parser.add_argument("--wake-inactivity-timeout", type=float, default=10.0, help="Seconds after TTS before requiring wake word again")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    mic_queue: Queue[bytes] = Queue()
    speaker_queue: Queue[bytes] = Queue(maxsize=128)
    stop_event = Event()
    wake_event: Event | None = None

    audio_streamer = AudioStreamer(
        mic_queue=mic_queue,
        speaker_queue=speaker_queue,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    streamer_thread = Thread(target=audio_streamer.run, args=(stop_event,), daemon=True)
    streamer_thread.start()

    if args.wake_word_model:
        wake_chime_bytes: bytes | None = None
        if args.wake_chime:
            wake_chime_bytes = _load_wav(args.wake_chime)
            if wake_chime_bytes:
                logger.info("Wake chime loaded from %s (%d bytes)", args.wake_chime, len(wake_chime_bytes))

        detector = WakeWordDetector(
            model_path=args.wake_word_model,
            threshold=args.wake_word_threshold,
            wake_chime_bytes=wake_chime_bytes,
        )
        wake_event = Event()
        detector.start(mic_queue, wake_event, speaker_queue=speaker_queue)
        logger.info("Wake word detection active, model=%s", args.wake_word_model)

    try:
        asyncio.run(
            websocket_client(
                host=args.host,
                port=args.port,
                mic_queue=mic_queue,
                speaker_queue=speaker_queue,
                audio_streamer=audio_streamer,
                stop_event=stop_event,
                wake_event=wake_event,
                wake_inactivity_timeout=args.wake_inactivity_timeout,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
