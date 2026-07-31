from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from faster_whisper import WhisperModel
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

console = Console()

logger = logging.getLogger(__name__)


class FasterWhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name: str = "tiny.en",
        device: str = "auto",
        compute_type: str = "auto",
        gen_kwargs: dict[str, Any] = {},
        _shared_stt_model: Any = None,
    ) -> None:
        self.gen_kwargs = self.adapt_gen_kwargs(gen_kwargs)

        if _shared_stt_model is not None:
            self.model = _shared_stt_model
            self._model_loaded_externally = True
            logger.info("FasterWhisperSTTHandler using shared STT model")
            return

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode != "final":
            return

        logger.debug("infering faster whisper...")

        segments, info = self.model.transcribe(vad_audio.audio, **self.gen_kwargs)
        output_text = []

        for segment in segments:
            logger.debug("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            output_text.append(segment.text)

        pred_text = " ".join(output_text).strip()

        logger.debug("finished whisper inference")
        if pred_text:
            console.print(f"[yellow]USER: {pred_text}")

            yield Transcription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
        else:
            logger.debug("no text detected. skipping...")

    def cleanup(self) -> None:
        print("Stopping FasterWhisperSTTHandler")
        if not getattr(self, "_model_loaded_externally", False):
            del self.model

    def adapt_gen_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        return_timestamps = gen_kwargs.pop("return_timestamps", False)
        if return_timestamps:
            gen_kwargs["word_timestamps"] = True
        else:
            gen_kwargs.pop("without_timestamps", None)

        return gen_kwargs
