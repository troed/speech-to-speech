from __future__ import annotations

import numpy as np

from speech_to_speech.pipeline.messages import (
    GenerateResponseRequest,
    Transcription,
    VADAudio,
)


class TestNativeLLMSTTHandler:
    def test_converts_float32_audio_to_int16_bytes(self):
        """Float32 ndarray audio → int16 PCM bytes."""
        from speech_to_speech.STT.native_llm_stt_handler import NativeLLMSTTHandler

        sr = 16000
        duration = 0.5
        samples = sr * duration
        audio_float32 = np.sin(2 * np.pi * 440 * np.arange(samples) / sr).astype(np.float32) * 0.5

        handler = object.__new__(NativeLLMSTTHandler)
        handler.setup()

        vad_audio = VADAudio(audio=audio_float32, mode="final")
        results = list(handler.process(vad_audio))

        assert len(results) == 1
        assert isinstance(results[0], Transcription)
        audio_bytes = results[0].audio_bytes
        assert audio_bytes is not None
        assert len(audio_bytes) == samples * 2  # 16-bit = 2 bytes per sample

        decoded = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert np.max(np.abs(decoded - audio_float32)) < 0.01

    def test_skips_progressive_mode(self):
        from speech_to_speech.STT.native_llm_stt_handler import NativeLLMSTTHandler

        handler = object.__new__(NativeLLMSTTHandler)
        handler.setup()

        audio = np.zeros(8000, dtype=np.float32)
        vad_audio = VADAudio(audio=audio, mode="progressive")

        results = list(handler.process(vad_audio))
        assert results == []

    def test_empty_audio_skips(self):
        from speech_to_speech.STT.native_llm_stt_handler import NativeLLMSTTHandler

        handler = object.__new__(NativeLLMSTTHandler)
        handler.setup()

        audio = np.array([], dtype=np.float32)
        vad_audio = VADAudio(audio=audio, mode="final")

        results = list(handler.process(vad_audio))
        assert results == []

    def test_passes_through_metadata(self):
        from speech_to_speech.STT.native_llm_stt_handler import NativeLLMSTTHandler

        handler = object.__new__(NativeLLMSTTHandler)
        handler.setup()

        audio = np.ones(8000, dtype=np.float32)
        vad_audio = VADAudio(
            audio=audio,
            mode="final",
            turn_id="turn_1",
            turn_revision=2,
            created_at_s=12345.0,
        )

        results = list(handler.process(vad_audio))
        assert len(results) == 1
        t = results[0]
        assert t.turn_id == "turn_1"
        assert t.turn_revision == 2
        assert t.speech_stopped_at_s == 12345.0
        assert t.text == ""
        assert t.audio_bytes is not None

    def test_shared_model_path_is_noop(self):
        """_shared_stt_model is accepted for pipeline wiring but not used."""
        from speech_to_speech.STT.native_llm_stt_handler import NativeLLMSTTHandler

        handler = object.__new__(NativeLLMSTTHandler)
        handler.setup(_shared_stt_model="fake_model")

        audio = np.ones(8000, dtype=np.float32)
        vad_audio = VADAudio(audio=audio, mode="final")
        results = list(handler.process(vad_audio))
        assert len(results) == 1


class TestTranscriptionNotifierAudioBytes:
    def test_passes_audio_bytes_to_event(self):
        from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier

        notifier = object.__new__(TranscriptionNotifier)
        notifier.setup(text_output_queue=None, runtime_config=None)
        notifier.echo_filter = None

        audio = b"\x00\x01\x02\x03"
        t = Transcription(text="hello", audio_bytes=audio, turn_id="t1", turn_revision=1, speech_stopped_at_s=123.0)

        results = list(notifier.process(t))
        assert len(results) == 0  # realtime mode yields nothing

    def test_passes_audio_bytes_to_event_with_output_queue(self):
        from queue import Queue

        from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier

        q: Queue = Queue()
        notifier = object.__new__(TranscriptionNotifier)
        notifier.setup(text_output_queue=q, runtime_config=None)
        notifier.echo_filter = None

        audio = b"\xaa\xbb\xcc"
        t = Transcription(text="hello", audio_bytes=audio, turn_id="t2", turn_revision=3)

        list(notifier.process(t))
        event = q.get_nowait()
        assert event.audio_bytes == audio


class TestChatCompletionsAudioSerialization:
    def test_serialize_injects_input_audio_as_wav(self):
        import base64
        import io
        import struct
        import wave
        import numpy as np

        from speech_to_speech.LLM.chat import Chat, make_user_message
        from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

        sr = 16000
        samples = sr  # 1 second
        pcm_audio = (
            (np.sin(2 * np.pi * 440 * np.arange(samples) / sr) * 0.5 * 32767)
            .astype(np.int16)
            .tobytes()
        )
        handler = object.__new__(ChatCompletionsApiModelHandler)
        handler._pending_audio = pcm_audio

        chat = Chat(30)
        chat.add_item(make_user_message("Hello"))

        messages = handler._serialize(chat)
        content = messages[0]["content"]
        audio_part = next(p for p in content if p["type"] == "input_audio")
        data = audio_part["input_audio"]["data"]
        assert audio_part["input_audio"]["format"] == "wav"

        wav_bytes = base64.b64decode(data)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            frames = wf.readframes(wf.getnframes())
            assert len(frames) == len(pcm_audio)

    def test_serialize_injects_input_audio(self):
        from speech_to_speech.LLM.chat import Chat, make_user_message
        from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

        handler = object.__new__(ChatCompletionsApiModelHandler)
        handler._pending_audio = b"\x00\x01"

        chat = Chat(30)
        chat.add_item(make_user_message("Hello"))

        messages = handler._serialize(chat)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) >= 2

        part_types = [p["type"] for p in content]
        assert "input_audio" in part_types
        audio_part = next(p for p in content if p["type"] == "input_audio")
        assert audio_part["input_audio"]["data"] is not None
        assert audio_part["input_audio"]["format"] == "wav"

    def test_serialize_without_audio_unchanged(self):
        from speech_to_speech.LLM.chat import Chat, make_user_message
        from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

        handler = object.__new__(ChatCompletionsApiModelHandler)
        handler._pending_audio = None

        chat = Chat(30)
        chat.add_item(make_user_message("Hello"))

        messages = handler._serialize(chat)
        content = messages[0]["content"]
        part_types = [p["type"] for p in content] if isinstance(content, list) else ["text"]
        assert "input_audio" not in part_types

    def test_audio_cleared_after_serialize(self):
        from speech_to_speech.LLM.chat import Chat, make_user_message
        from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

        handler = object.__new__(ChatCompletionsApiModelHandler)
        handler._pending_audio = b"test"

        chat = Chat(30)
        chat.add_item(make_user_message("Hello"))

        handler._serialize(chat)
        assert handler._pending_audio is None


class TestTranscriptionWithAudioBytes:
    def test_transcription_has_audio_bytes_field(self):
        t = Transcription(text="hello", audio_bytes=b"\x00\x01\x02")
        assert t.audio_bytes == b"\x00\x01\x02"

    def test_transcription_audio_bytes_defaults_to_none(self):
        t = Transcription(text="hello")
        assert t.audio_bytes is None


class TestServiceAddsUserMessageForAudioOnly:
    """When transcript is empty but audio_bytes is present (native-llm STT),
    _on_transcription_completed must still add a user message to the chat."""

    def test_chat_has_user_message_after_transcription_completed_with_audio_only(self):
        from speech_to_speech.api.openai_realtime.runtime_config import RealtimeSessionCreateRequest, RuntimeConfig
        from speech_to_speech.LLM.chat import Chat, make_user_message
        from speech_to_speech.pipeline.events import TranscriptionCompletedEvent

        session = RealtimeSessionCreateRequest(type="realtime")
        cfg = RuntimeConfig(chat=Chat(30), session=session)

        audio = b"\x00\x01\x02"
        event = TranscriptionCompletedEvent(
            transcript="",
            audio_bytes=audio,
            turn_id="t1",
            turn_revision=0,
            speech_stopped_at_s=100.0,
        )
        transcript = event.transcript or ("[audio]" if event.audio_bytes else "")

        cfg.chat.add_item(make_user_message(transcript))

        messages = cfg.chat.to_transformers_chat()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_no_audio_no_transcript_skips_user_message(self):
        from speech_to_speech.pipeline.events import TranscriptionCompletedEvent

        event = TranscriptionCompletedEvent(
            transcript="",
            audio_bytes=None,
            turn_id="t2",
            turn_revision=0,
            speech_stopped_at_s=200.0,
        )
        transcript = event.transcript
        audio_bytes = event.audio_bytes
        assert not (transcript or audio_bytes)


class TestGenerateResponseRequestWithAudioBytes:
    def test_request_has_audio_bytes_field(self):
        from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

        cfg = RuntimeConfig()
        req = GenerateResponseRequest(runtime_config=cfg, audio_bytes=b"test")
        assert req.audio_bytes == b"test"

    def test_request_audio_bytes_defaults_to_none(self):
        from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

        cfg = RuntimeConfig()
        req = GenerateResponseRequest(runtime_config=cfg)
        assert req.audio_bytes is None
