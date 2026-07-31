from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)


class NativeLLMSTTHandler(BaseSTTHandler):
    """Pass-through STT handler that packages VAD audio as int16 PCM bytes.

    Instead of running a local speech-to-text model, this handler converts the
    audio segment to raw 16-bit PCM (Little-endian, 16 kHz mono) and attaches it
    to the ``Transcription`` message so the LLM handler can send it as an
    ``input_audio`` part to a multimodal Chat Completions API (e.g. Gemma 4
    12B Unified).
    """

    def setup(
        self,
        _shared_stt_model: Any = None,
        **_kwargs: Any,
    ) -> None:
        pass

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode != "final":
            return

        array = vad_audio.audio
        if array is None or len(array) == 0:
            return

        if isinstance(array, np.ndarray) and array.dtype == np.float32:
            audio_int16 = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
        elif isinstance(array, np.ndarray) and array.dtype == np.int16:
            audio_bytes = array.tobytes()
        else:
            audio_bytes = np.asarray(array, dtype=np.int16).tobytes()

        logger.debug("NativeLLMSTTHandler: packaged %d bytes of audio", len(audio_bytes))

        yield Transcription(
            text="",
            audio_bytes=audio_bytes,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
