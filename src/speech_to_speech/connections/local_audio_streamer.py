import logging
import threading
import time
from collections import deque
from queue import Queue

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: threading.Event,
        list_play_chunk_size: int = 512,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen
        self._pcm_buffer = deque()

    def _next_pcm_frame(self, frames: int) -> np.ndarray:
        """Return ``frames`` samples of int16 PCM from the internal buffer, consuming from it."""
        out = np.empty(frames, dtype=np.int16)
        written = 0
        while written < frames and self._pcm_buffer:
            chunk = self._pcm_buffer[0]
            needed = frames - written
            if len(chunk) <= needed:
                self._pcm_buffer.popleft()
                out[written:written + len(chunk)] = chunk
                written += len(chunk)
            else:
                out[written:] = chunk[:needed]
                self._pcm_buffer[0] = chunk[needed:]
                written = frames
        if written < frames:
            out[written:] = 0  # pad with silence
        return out

    def _consume_chunk(self, audio_chunk: AudioOutItem) -> None:
        """Push an output queue item into the PCM buffer."""
        if isinstance(audio_chunk, np.ndarray):
            pcm = audio_chunk.ravel().astype(np.int16)
        elif isinstance(audio_chunk, bytes):
            pcm = np.frombuffer(audio_chunk, dtype=np.int16)
        elif isinstance(audio_chunk, AudioOutput):
            inner = audio_chunk.audio
            pcm = inner.ravel().astype(np.int16) if isinstance(inner, np.ndarray) else np.frombuffer(inner, dtype=np.int16)
        else:
            pcm = np.array([], dtype=np.int16)
        if len(pcm):
            # Split into 512-sample chunks for the callback
            for i in range(0, len(pcm), self.list_play_chunk_size):
                self._pcm_buffer.append(pcm[i:i + self.list_play_chunk_size])

    def run(self) -> None:
        dither = np.random.randint(-1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16)
        _dummy_silence = np.zeros(self.list_play_chunk_size, dtype=np.int16)

        def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time: float, status: str) -> None:
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            # Drain any new items from the output queue into the PCM buffer
            while not self.output_queue.empty():
                try:
                    chunk = self.output_queue.get_nowait()
                    if chunk is AUDIO_RESPONSE_DONE:
                        self.should_listen.set()
                        logger.debug("Response complete, listening re-enabled")
                    else:
                        self._consume_chunk(chunk)
                except Exception:
                    pass

            if self._pcm_buffer:
                outdata[:] = self._next_pcm_frame(frames)[:, np.newaxis]
            else:
                pcm = np.ascontiguousarray(indata, dtype=np.int16)
                self.input_queue.put(pcm.tobytes())
                outdata[:] = dither

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
