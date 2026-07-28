from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from queue import Queue
from threading import Event
from typing import TYPE_CHECKING, Optional

import numpy as np

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
        wake_chime_bytes: Optional[bytes] = None,
        chime_output_queue: Optional[Queue] = None,
        should_listen: Optional[Event] = None,
    ) -> None:
        self._threshold = threshold
        self._activation_timeout_s = activation_timeout_s
        self._state = "sleeping"
        self._buffer: list[bytes] = []
        self._last_forward_time: float = 0.0
        self._cooldown_until: float = 0.0
        self._wake_chime_bytes = wake_chime_bytes
        self._chime_output_queue = chime_output_queue
        self._should_listen = should_listen
        sample_rate = 16000
        self._preroll_chunks = max(1, int((preroll_ms / 1000) * sample_rate / 512))

        try:
            from openwakeword.model import Model

            self._model = Model(wakeword_model_paths=[model_path])
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
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            prediction = self._model.predict(audio_array)
            if prediction and max(prediction.values()) >= self._threshold:
                now = time.monotonic()
                if self._should_listen is not None and not self._should_listen.is_set():
                    logger.info(
                        "WakeWordHandler: detection suppressed (should_listen cleared), "
                        "stale=%.0fms", (now - self._last_forward_time) * 1000
                    )
                elif now < self._cooldown_until:
                    logger.info(
                        "WakeWordHandler: detection suppressed (cooldown %.0fms remaining)",
                        (self._cooldown_until - now) * 1000,
                    )
                else:
                    self._state = "active"
                    self._last_forward_time = now
                    self._play_wake_chime()
                    for b in self._buffer:
                        yield (b, rt_cfg) if rt_cfg else b
                    yield item
                    self._buffer.clear()
                    logger.info("WakeWordHandler: wake word detected, activating")
                    return
            self._buffer.append(chunk)
            while len(self._buffer) > self._preroll_chunks:
                self._buffer.pop(0)

        elif self._state == "active":
            now = time.monotonic()
            if now - self._last_forward_time > self._activation_timeout_s:
                self._state = "sleeping"
                self._buffer.clear()
                self._cooldown_until = now + 2.0
                logger.info("WakeWordHandler: activation timeout expired, going to sleep (cooldown 2s)")
                return
            # Only extend the activation window for voice-level audio, not silence
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            if np.abs(audio_array.astype(np.int32)).max() >= 200:
                self._last_forward_time = now
            yield item

    def _play_wake_chime(self) -> None:
        if self._wake_chime_bytes is not None and self._chime_output_queue is not None:
            logger.info("WakeWordHandler: playing wake chime (%d bytes)", len(self._wake_chime_bytes))
            try:
                self._chime_output_queue.put_nowait(self._wake_chime_bytes)
            except Exception as exc:
                logger.warning("WakeWordHandler: failed to queue wake chime: %s", exc)
        else:
            logger.info(
                "WakeWordHandler: chime not played (bytes=%s, queue=%s)",
                self._wake_chime_bytes is not None,
                self._chime_output_queue is not None,
            )

    def on_session_end(self) -> None:
        self._state = "sleeping"
        self._buffer.clear()
