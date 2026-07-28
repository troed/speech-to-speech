from __future__ import annotations

import logging
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PIPELINE_RATE = 16000


def _load_wav(path: str) -> Optional[bytes]:
    try:
        with wave.open(path, "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            frames = w.readframes(w.getnframes())
    except Exception as e:
        logger.warning("ChimeLoader: failed to read WAV %s: %s", path, e)
        return None

    if sampwidth != 2:
        logger.warning("ChimeLoader: unsupported sample width %d for %s (must be 16-bit)", sampwidth, path)
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


class ChimeLoader:
    def __init__(self, wake_chime_path: str | None = None, search_chime_path: str | None = None) -> None:
        self._wake_chime: bytes | None = None
        self._search_chime: bytes | None = None
        if wake_chime_path:
            self._wake_chime = _load_wav(wake_chime_path)
        if search_chime_path:
            self._search_chime = _load_wav(search_chime_path)

    @property
    def wake_chime(self) -> bytes | None:
        return self._wake_chime

    @property
    def search_chime(self) -> bytes | None:
        return self._search_chime
