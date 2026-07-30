import logging
import threading
import time
from collections import deque
from queue import Queue

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: threading.Event,
        response_done_event: threading.Event | None = None,
        response_playing: threading.Event | None = None,
        echo_reference_queue: Queue | None = None,
        list_play_chunk_size: int = 512,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen
        self._response_done_event = response_done_event
        self._response_playing = response_playing
        self._echo_reference_queue = echo_reference_queue
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
            if self._response_playing is not None and not self._response_playing.is_set():
                self._response_playing.set()
            for i in range(0, len(pcm), self.list_play_chunk_size):
                chunk = pcm[i:i + self.list_play_chunk_size]
                self._pcm_buffer.append(chunk)
                if self._echo_reference_queue is not None:
                    try:
                        self._echo_reference_queue.put_nowait(chunk.tobytes())
                    except Exception:
                        pass

    def run(self) -> None:
        dither = np.random.randint(-1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16)

        def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time: float, status: str) -> None:
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            # Always capture mic input (full-duplex)
            pcm = np.ascontiguousarray(indata, dtype=np.int16)
            self.input_queue.put(pcm.tobytes())

            # Drain any new items from the output queue into the PCM buffer
            while not self.output_queue.empty():
                try:
                    chunk = self.output_queue.get_nowait()
                    if chunk is AUDIO_RESPONSE_DONE:
                        self.should_listen.set()
                        if self._response_done_event is not None:
                            self._response_done_event.set()
                        if self._response_playing is not None:
                            self._response_playing.clear()
                        logger.debug("Response complete, listening re-enabled")
                    elif isinstance(chunk, PipelineControlMessage) and is_control_message(chunk, SESSION_END.kind):
                        if self._response_playing is not None:
                            self._response_playing.clear()
                    else:
                        self._consume_chunk(chunk)
                except Exception:
                    pass

            if self._pcm_buffer:
                outdata[:] = self._next_pcm_frame(frames)[:, np.newaxis]
            else:
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
