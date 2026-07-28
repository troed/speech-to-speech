from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from queue import Queue
from threading import Event
from typing import TYPE_CHECKING

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
    ) -> None:
        self._threshold = threshold
        self._activation_timeout_s = activation_timeout_s
        self._state = "sleeping"
        self._buffer: list[bytes] = []
        self._last_forward_time: float = 0.0
        sample_rate = 16000
        self._preroll_chunks = max(1, int((preroll_ms / 1000) * sample_rate / 512))

        try:
            from openwakeword.model import Model

            self._model = Model(wakeword_models=[model_path])
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
            prediction = self._model.predict(chunk)
            if prediction and max(prediction.values()) >= self._threshold:
                self._state = "active"
                self._last_forward_time = time.monotonic()
                for b in self._buffer:
                    yield (b, rt_cfg) if rt_cfg else b
                yield item
                self._buffer.clear()
                logger.info("WakeWordHandler: wake word detected, activating")
            else:
                self._buffer.append(chunk)
                while len(self._buffer) > self._preroll_chunks:
                    self._buffer.pop(0)

        elif self._state == "active":
            now = time.monotonic()
            if now - self._last_forward_time > self._activation_timeout_s:
                self._state = "sleeping"
                self._buffer.clear()
                logger.info("WakeWordHandler: activation timeout expired, going to sleep")
                return
            self._last_forward_time = now
            yield item

    def on_session_end(self) -> None:
        self._state = "sleeping"
        self._buffer.clear()
