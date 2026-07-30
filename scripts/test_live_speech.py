"""
Send real speech audio to a running server and check for a response.

Usage:
  python scripts/test_live_speech.py                    # 127.0.0.1:8766
  python scripts/test_live_speech.py --host X --port Y
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import wave
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_live_speech")

PIPELINE_RATE = 16000
CHUNK_SAMPLES = 512
CHUNK_BYTES = CHUNK_SAMPLES * 2


def load_wav(path: str) -> bytes | None:
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
        logger.warning("Unsupported sample width %d", sampwidth)
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


async def test_speech(
    host: str,
    port: int,
    wav_path: str,
    repeat: int = 1,
    gap_s: float = 2.0,
) -> None:
    """Send speech WAV audio and check for server response."""
    import websockets.exceptions
    from websockets.asyncio.client import connect

    pcm_bytes = load_wav(wav_path)
    if pcm_bytes is None:
        logger.error("Failed to load WAV: %s", wav_path)
        return

    # Pad with silence to trigger VAD properly
    silence = b"\x00\x00" * CHUNK_SAMPLES

    logger.info("Loaded %s: %.1fs at 16kHz mono", wav_path, len(pcm_bytes) / 32000)
    logger.info("Connecting to ws://%s:%d ...", host, port)

    url = f"ws://{host}:{port}"
    received_audio = 0
    received_events: list[dict] = []

    try:
        async with connect(url) as ws:
            logger.info("Connected")

            # Send audio in real-time chunks (512 samples per ~32ms)
            # with leading silence to let VAD settle
            chunks_to_send = []
            # Leading silence
            for _ in range(10):
                chunks_to_send.append(silence)

            for _ in range(repeat):
                offset = 0
                while offset < len(pcm_bytes):
                    chunk = pcm_bytes[offset : offset + CHUNK_BYTES]
                    if len(chunk) < CHUNK_BYTES:
                        chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
                    chunks_to_send.append(chunk)
                    offset += CHUNK_BYTES
                # Trailing silence between repeats
                for _ in range(int(gap_s * 31)):  # ~31 chunks per second
                    chunks_to_send.append(silence)

            total_chunks = len(chunks_to_send)
            total_bytes = total_chunks * CHUNK_BYTES
            logger.info("Sending %d chunks (%d bytes) ...", total_chunks, total_bytes)

            async def send_loop():
                for chunk in chunks_to_send:
                    await ws.send(chunk)
                    await asyncio.sleep(0.03)  # ~real-time pace
                logger.info("Finished sending all chunks")

            async def recv_loop():
                nonlocal received_audio, received_events
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Connection closed by server")
                        break

                    if isinstance(msg, bytes):
                        received_audio += len(msg)
                        logger.info("RECV audio: %d bytes (total: %d)", len(msg), received_audio)
                    else:
                        try:
                            event = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        received_events.append(event)
                        logger.info("RECV event: %s", json.dumps(event, indent=2))

            tasks = [
                asyncio.create_task(send_loop()),
                asyncio.create_task(recv_loop()),
            ]

            # Wait for send to finish, then give server some time to respond
            await asyncio.wait_for(tasks[0], timeout=60.0)
            logger.info("Send complete, waiting for response...")
            await asyncio.sleep(10.0)

            tasks[1].cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        logger.error("Error: %s", e)

    logger.info("\n=== RESULTS ===")
    logger.info("Audio sent: %d chunks (%d bytes)", total_chunks, total_bytes)
    logger.info("Audio received: %d bytes", received_audio)
    logger.info("Events received: %d", len(received_events))
    for ev in received_events:
        logger.info("  - tag=%s", ev.get("tag", ev))

    if received_audio > 0 or received_events:
        logger.info("PASS: Server responded!")
    else:
        logger.info("FAIL: No response from server")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--wav", default="computer.wav", help="Path to WAV file")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to repeat")
    parser.add_argument("--gap", type=float, default=2.0, help="Gap between repeats (seconds)")
    args = parser.parse_args()

    asyncio.run(test_speech(args.host, args.port, args.wav, args.repeat, args.gap))
