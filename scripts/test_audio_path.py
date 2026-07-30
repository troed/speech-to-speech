"""
Reproduce the client wake-word audio path issue.

Usage:
  python scripts/test_audio_path.py                   # self-contained local test
  python scripts/test_audio_path.py --live             # connect to real server on port 8766
  python scripts/test_audio_path.py --live --host X    # custom host
  python scripts/test_audio_path.py --live --port 8765 # custom port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from queue import Empty, Queue
from threading import Event

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_audio_path")


async def run_local_server(host: str, port: int) -> tuple[asyncio.Server, list[bytes]]:
    """Start a WebSocket server that records received binary messages."""
    import websockets
    from websockets.asyncio.server import ServerConnection, serve

    received: list[bytes] = []

    async def handler(websocket: ServerConnection) -> None:
        logger.info("TEST_SERVER: Client connected")
        async for message in websocket:
            if isinstance(message, bytes):
                received.append(message)
                logger.info("TEST_SERVER: Received %d bytes (total chunks: %d)", len(message), len(received))
            else:
                data = json.loads(message)
                logger.info("TEST_SERVER: Received text: %s", data.get("tag", ""))

    server = await serve(handler, host, port)
    logger.info("TEST_SERVER: Listening on ws://%s:%d", host, port)
    return server, received


async def client_send_audio(
    host: str,
    port: int,
    wake_event: Event | None = None,
    test_audio: bytes = b"",
    send_duration: float = 2.0,
) -> dict:
    """Replicate the client's audio-sending logic after wake word fires."""
    import websockets.exceptions
    from websockets.asyncio.client import connect

    url = f"ws://{host}:{port}"
    result = {
        "connected": False,
        "sent_chunks": 0,
        "sent_bytes": 0,
        "received_responses": 0,
        "errors": [],
    }

    mic_queue: Queue = Queue()
    stop_event = Event()
    in_conversation = False
    activation_time = 0.0

    async def fake_audio_source():
        nonlocal in_conversation
        if test_audio:
            for _ in range(5):
                mic_queue.put(test_audio)
                await asyncio.sleep(0.01)
        while not stop_event.is_set():
            if wake_event is None or wake_event.is_set():
                if test_audio:
                    mic_queue.put(test_audio)
            await asyncio.sleep(0.05)

    async def send_audio(ws):
        nonlocal in_conversation, activation_time
        while not stop_event.is_set():
            try:
                chunk = await asyncio.to_thread(mic_queue.get, True, 0.1)
            except Empty:
                continue

            if wake_event is not None:
                if wake_event.is_set():
                    if not in_conversation:
                        in_conversation = True
                        activation_time = time.monotonic()
                        logger.info("CLIENT: Conversation active, streaming audio")
                    await ws.send(chunk)
                    result["sent_chunks"] += 1
                    result["sent_bytes"] += len(chunk)
                else:
                    continue
            else:
                await ws.send(chunk)
                result["sent_chunks"] += 1
                result["sent_bytes"] += len(chunk)

    async def receive_audio(ws):
        while not stop_event.is_set():
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("CLIENT: Server disconnected")
                stop_event.set()
                break

            if isinstance(message, bytes):
                result["received_responses"] += 1
                logger.info("CLIENT: Received %d bytes of audio", len(message))
            else:
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                logger.info("CLIENT: Received text event: tag=%s", event.get("tag", ""))

    try:
        logger.info("CLIENT: Connecting to %s ...", url)
        async with connect(url) as ws:
            result["connected"] = True
            logger.info("CLIENT: Connected")

            source_task = asyncio.create_task(fake_audio_source())

            if wake_event is not None:
                logger.info("CLIENT: Waiting for wake word...")
                while not wake_event.is_set() and not stop_event.is_set():
                    await asyncio.sleep(0.1)
                if stop_event.is_set():
                    source_task.cancel()
                    return result

            tasks = [
                asyncio.create_task(send_audio(ws)),
                asyncio.create_task(receive_audio(ws)),
            ]

            if wake_event is None:
                await asyncio.sleep(send_duration + 2.0)
            else:
                await asyncio.sleep(send_duration)

            stop_event.set()
            for t in tasks:
                t.cancel()
            source_task.cancel()
    except Exception as e:
        logger.error("CLIENT: Error: %s", e)
        result["errors"].append(str(e))
    finally:
        stop_event.set()

    return result


def test_local():
    """Self-contained test with local server."""
    port = 18765

    async def _run():
        server, received = await run_local_server("127.0.0.1", port)
        await asyncio.sleep(0.1)

        wake_event = Event()
        test_chunk = b"\x00\x00" * 256

        client_task = asyncio.create_task(
            client_send_audio("127.0.0.1", port, wake_event=wake_event, test_audio=test_chunk, send_duration=2.0)
        )

        await asyncio.sleep(0.5)
        logger.info("\n=== TEST: Firing wake word ===")
        wake_event.set()
        await asyncio.sleep(1.0)

        result = await client_task
        server.close()
        await server.wait_closed()

        logger.info("\n=== LOCAL TEST RESULTS ===")
        logger.info("Connected: %s", result["connected"])
        logger.info("Sent chunks: %d", result["sent_chunks"])
        logger.info("Sent bytes: %d", result["sent_bytes"])
        logger.info("Server received: %d chunks (%d bytes)", len(received), sum(len(c) for c in received))
        logger.info("Errors: %s", result["errors"])

        if result["sent_chunks"] > 0 and len(received) > 0:
            logger.info("PASS: Audio path works")
        else:
            logger.info("FAIL: Audio did not flow")

    asyncio.run(_run())


def test_live(host: str, port: int):
    """Connect to the user's running server and send test audio."""

    sample_rate = 16000
    t = np.linspace(0, 1.0, int(sample_rate * 1.0), endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    test_chunk = sine.tobytes()

    async def test_no_wake():
        """Test 1: Connect without wake word, send audio, wait for response."""
        logger.info("\n=== TEST 1: No wake word, send audio ===")
        result = await client_send_audio(host, port, wake_event=None, test_audio=test_chunk, send_duration=5.0)
        logger.info("\n=== TEST 1 RESULTS ===")
        logger.info("Connected: %s", result["connected"])
        logger.info("Sent chunks: %d (%d bytes)", result["sent_chunks"], result["sent_bytes"])
        logger.info("Received responses: %d", result["received_responses"])
        logger.info("Errors: %s", result["errors"])
        if result["received_responses"] > 0:
            logger.info("PASS: Server responded to audio")
        else:
            logger.info("INCONCLUSIVE: No response (may need longer audio or VAD threshold)")
        return result

    async def test_with_wake():
        """Test 2: Connect with wake word, fire it, send audio."""
        logger.info("\n=== TEST 2: With wake word ===")
        wake_event = Event()
        result = await client_send_audio(host, port, wake_event=wake_event, test_audio=test_chunk, send_duration=5.0)
        logger.info("\n=== TEST 2 RESULTS ===")
        logger.info("Connected: %s", result["connected"])
        logger.info("Sent chunks: %d (%d bytes)", result["sent_chunks"], result["sent_bytes"])
        logger.info("Received responses: %d", result["received_responses"])
        logger.info("Errors: %s", result["errors"])
        if result["received_responses"] > 0:
            logger.info("PASS: Server responded")
        elif result["sent_chunks"] > 0:
            logger.info("Audio reached server but no response (VAD/STT may not process sine wave)")
        else:
            logger.info("FAIL: No audio sent despite wake word")
        return result

    asyncio.run(test_no_wake())
    logger.info("\n" + "=" * 60)
    asyncio.run(test_with_wake())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Connect to real server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if args.live:
        test_live(args.host, args.port)
    else:
        test_local()
