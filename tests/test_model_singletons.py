import threading
from queue import Queue
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
from speech_to_speech.VAD.vad_handler import VADHandler


def _stream_item(audio=None, sample_rate=24000):
    """Create a TTS stream item compatible with _prepare_audio_chunk."""
    if audio is None:
        audio = np.full(512, 0.1, dtype=np.float32)
    return (audio, sample_rate, 0.0)


class TestFasterWhisperSTTHandlerSharedModel:

    def test_setup_uses_shared_model_when_provided(self, monkeypatch):
        from speech_to_speech.STT import faster_whisper_handler as stt_mod

        shared = MagicMock()
        monkeypatch.setattr(stt_mod, "WhisperModel", MagicMock())

        handler = object.__new__(FasterWhisperSTTHandler)
        handler.setup(
            model_name="tiny.en",
            device="auto",
            compute_type="auto",
            _shared_stt_model=shared,
        )

        assert handler.model is shared
        assert handler._model_loaded_externally is True
        stt_mod.WhisperModel.assert_not_called()

    def test_setup_loads_model_when_no_shared(self, monkeypatch):
        from speech_to_speech.STT import faster_whisper_handler as stt_mod

        mock_whisper = MagicMock()
        monkeypatch.setattr(stt_mod, "WhisperModel", mock_whisper)

        handler = object.__new__(FasterWhisperSTTHandler)
        handler.setup(
            model_name="tiny.en",
            device="cpu",
            compute_type="int8",
        )

        assert handler.model is mock_whisper.return_value
        assert not getattr(handler, "_model_loaded_externally", False)
        mock_whisper.assert_called_once_with("tiny.en", device="cpu", compute_type="int8")

    def test_cleanup_skips_deletion_for_shared_model(self):
        handler = object.__new__(FasterWhisperSTTHandler)
        handler.model = MagicMock()
        handler._model_loaded_externally = True

        handler.cleanup()

    def test_cleanup_deletes_own_model(self):
        handler = object.__new__(FasterWhisperSTTHandler)
        handler.model = MagicMock()

        handler.cleanup()

        with pytest.raises(AttributeError):
            _ = handler.model


class TestQwen3TTSHandlerSharedModel:

    def test_setup_uses_shared_model_and_lock(self, monkeypatch):
        monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
        monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

        shared_model = MagicMock()
        shared_lock = threading.Lock()

        handler = object.__new__(Qwen3TTSHandler)
        handler.setup(
            Event(),
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device="cuda",
            backend="ggml",
            _shared_tts_model=shared_model,
            _shared_tts_lock=shared_lock,
        )

        assert handler.model is shared_model
        assert handler._model_lock is shared_lock
        assert handler._model_loaded_externally is True
        assert handler.backend == "faster_qwen3_tts"

    def test_setup_skips_warmup_for_shared_model(self, monkeypatch):
        monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
        warmup_mock = MagicMock()
        monkeypatch.setattr(Qwen3TTSHandler, "warmup", warmup_mock)

        handler = object.__new__(Qwen3TTSHandler)
        handler.setup(
            Event(),
            _shared_tts_model=MagicMock(),
            _shared_tts_lock=threading.Lock(),
        )

        warmup_mock.assert_not_called()

    def test_setup_loads_fresh_model_when_no_shared(self, monkeypatch):
        monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
        recorded = {}

        def _setup_faster(self, **kwargs):
            recorded.update(kwargs)
            self.model = MagicMock()

        monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
        monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

        handler = object.__new__(Qwen3TTSHandler)
        handler.setup(
            Event(),
            model_name="my-model",
            device="cuda",
            backend="ggml",
            quant="Q8_0",
        )

        assert not getattr(handler, "_model_loaded_externally", False)
        assert recorded.get("model_name") == "my-model"
        assert recorded.get("quant") == "Q8_0"

    def test_cleanup_skips_deletion_for_shared_model(self):
        handler = object.__new__(Qwen3TTSHandler)
        handler.model = MagicMock()
        handler._model_loaded_externally = True
        handler._mlx_temp_ref_audio_files = set()
        handler.backend = "faster_qwen3_tts"

        handler.cleanup()
        assert hasattr(handler, "model")

    def test_cleanup_deletes_own_model(self):
        handler = object.__new__(Qwen3TTSHandler)
        handler.model = MagicMock()
        handler._mlx_temp_ref_audio_files = set()
        handler.backend = "faster_qwen3_tts"

        handler.cleanup()

        with pytest.raises(AttributeError):
            _ = handler.model

    def test_stream_acquires_lock_before_generation(self):
        """_stream() must acquire the shared lock before iterating the model."""
        handler = object.__new__(Qwen3TTSHandler)
        handler.cancel_scope = None
        handler.blocksize = 512

        lock = threading.Lock()
        initial = lock.locked()

        gen = iter([_stream_item()])
        list(handler._stream(gen, label="test", lock=lock))

        assert not initial
        assert not lock.locked()

    def test_stream_releases_lock_on_exception(self):
        """_stream() must release the lock even when the generator raises."""
        handler = object.__new__(Qwen3TTSHandler)
        handler.cancel_scope = None
        handler.blocksize = 512

        lock = threading.Lock()

        def broken_gen():
            yield _stream_item()
            raise RuntimeError("injected failure")

        try:
            list(handler._stream(broken_gen(), label="test", lock=lock))
        except RuntimeError:
            pass

        assert not lock.locked()

    def test_stream_without_lock_never_locks(self):
        """_stream() must work without a lock and not affect any."""
        handler = object.__new__(Qwen3TTSHandler)
        handler.cancel_scope = None
        handler.blocksize = 512

        gen = iter([_stream_item() for _ in range(3)])

        chunks = list(handler._stream(gen, label="test"))

        assert len(chunks) == 3

    def test_process_voice_clone_passes_lock_to_stream(self, monkeypatch):
        monkeypatch.setattr(qwen3_tts_module, "platform", "linux")

        handler = object.__new__(Qwen3TTSHandler)
        handler.model = MagicMock()
        handler.model.generate_voice_clone_streaming.return_value = iter([_stream_item()])
        handler.backend = "faster_qwen3_tts"
        handler.faster_backend = "ggml"
        handler.streaming_chunk_size = 4
        handler.language = "en"
        handler.ref_audio = "/tmp/test.wav"
        handler.ref_text = "test text"
        handler.xvec_only = False
        handler.parity_mode = False
        handler.non_streaming_mode = True
        handler.cancel_scope = None
        handler._model_lock = threading.Lock()
        handler.blocksize = 512

        stream_calls = []

        def fake_stream(gen, label, lock=None):
            stream_calls.append(lock)
            return iter([])

        monkeypatch.setattr(handler, "_stream", fake_stream)
        monkeypatch.setattr(handler, "_estimate_max_new_tokens", lambda text: 100)

        list(handler._process_voice_clone("hello"))

        assert stream_calls[0] is handler._model_lock


class TestVADHandlerSharedModel:

    def test_setup_uses_shared_model(self, monkeypatch):
        import speech_to_speech.VAD.vad_handler as vad_mod

        shared = MagicMock()
        monkeypatch.setattr(vad_mod, "VADIterator", MagicMock())

        handler = object.__new__(VADHandler)
        handler.setup(
            Event(),
            _shared_vad_model=shared,
        )

        assert handler.model is shared

    def test_setup_loads_model_when_no_shared(self, monkeypatch):
        import speech_to_speech.VAD.vad_handler as vad_mod

        monkeypatch.setattr(vad_mod, "VADIterator", MagicMock())

        calls = []

        def fake_load(*args, **kwargs):
            calls.append(args)
            return MagicMock(), MagicMock()

        monkeypatch.setattr(vad_mod.torch.hub, "load", fake_load)

        handler = object.__new__(VADHandler)
        handler.setup(Event())

        assert len(calls) == 1
        assert calls[0][0] == "snakers4/silero-vad"


class TestEnsureModelSingletons:

    def test_returns_dict_with_expected_keys(self, monkeypatch):
        from speech_to_speech.s2s_pipeline import _ensure_model_singletons

        fake_whisper = MagicMock()
        fake_tts = MagicMock()
        fake_tts_model = MagicMock()
        fake_tts.from_pretrained.return_value = fake_tts_model
        fake_tts_model.warmup = MagicMock()

        monkeypatch.setattr("faster_whisper.WhisperModel", fake_whisper)
        monkeypatch.setattr("faster_qwen3_tts.FasterQwen3TTS", fake_tts)
        monkeypatch.setattr("torch.hub.load",
                            lambda *a, **kw: (MagicMock(), MagicMock()))

        import speech_to_speech.s2s_pipeline as s2s
        s2s._shared_models_initialized = False
        s2s._shared_stt_model = None
        s2s._shared_tts_model = None
        s2s._shared_vad_model = None

        fake_args = MagicMock()
        fake_args.model_name = "tiny.en"
        fake_args.device = "cpu"
        fake_args.compute_type = "int8"

        tts_args = MagicMock()
        tts_args.model_name = "test-model"
        tts_args.device = "cpu"
        tts_args.dtype = "auto"
        tts_args.attn_implementation = "eager"
        tts_args.backend = "ggml"
        tts_args.quant = None

        result = _ensure_model_singletons(fake_args, tts_args)

        assert "stt_model" in result
        assert "tts_model" in result
        assert "tts_lock" in result
        assert "vad_model" in result
        assert result["stt_model"] is fake_whisper.return_value
        assert result["tts_model"] is fake_tts_model
        assert result["tts_lock"] is not None

    def test_singletons_loaded_only_once(self, monkeypatch):
        from speech_to_speech.s2s_pipeline import _ensure_model_singletons

        whis_call_count = 0

        class CountingWhisper:
            def __init__(self, *args, **kwargs):
                nonlocal whis_call_count
                whis_call_count += 1

        fake_tts = MagicMock()
        fake_tts_model = MagicMock()
        fake_tts.from_pretrained.return_value = fake_tts_model
        fake_tts_model.warmup = MagicMock()

        monkeypatch.setattr("faster_whisper.WhisperModel", CountingWhisper)
        monkeypatch.setattr("faster_qwen3_tts.FasterQwen3TTS", fake_tts)
        monkeypatch.setattr("torch.hub.load",
                            lambda *a, **kw: (MagicMock(), MagicMock()))

        import speech_to_speech.s2s_pipeline as s2s
        s2s._shared_models_initialized = False
        s2s._shared_stt_model = None
        s2s._shared_tts_model = None
        s2s._shared_vad_model = None

        fake_args = MagicMock()
        fake_args.model_name = "tiny.en"
        fake_args.device = "cpu"
        fake_args.compute_type = "int8"

        tts_args = MagicMock()
        tts_args.model_name = "test-model"
        tts_args.device = "cpu"
        tts_args.dtype = "auto"
        tts_args.attn_implementation = "eager"
        tts_args.backend = "ggml"
        tts_args.quant = None

        r1 = _ensure_model_singletons(fake_args, tts_args)
        r2 = _ensure_model_singletons(fake_args, tts_args)

        assert whis_call_count == 1
        assert r1["stt_model"] is r2["stt_model"]
        assert r1["tts_model"] is r2["tts_model"]
        assert r1["tts_lock"] is r2["tts_lock"]
        assert r1["vad_model"] is r2["vad_model"]
