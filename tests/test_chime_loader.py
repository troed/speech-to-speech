from __future__ import annotations

import wave

import numpy as np
import pytest

from speech_to_speech.chime_loader import ChimeLoader, _load_wav


def _make_wav(path: str, samples: np.ndarray, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())


def test_load_wav_returns_bytes(tmp_path):
    p = str(tmp_path / "test.wav")
    samples = np.arange(16000, dtype=np.int16)
    _make_wav(p, samples, 16000)
    result = _load_wav(p)
    assert isinstance(result, bytes)
    assert len(result) == 16000 * 2


def test_load_wav_missing_file_returns_none(tmp_path):
    result = _load_wav(str(tmp_path / "nonexistent.wav"))
    assert result is None


def test_load_wav_invalid_file_returns_none(tmp_path):
    p = str(tmp_path / "bad.wav")
    with open(p, "wb") as f:
        f.write(b"not a wav file")
    result = _load_wav(p)
    assert result is None


def test_load_wav_resamples_48khz(tmp_path):
    p = str(tmp_path / "test.wav")
    samples = np.arange(48000, dtype=np.int16)
    _make_wav(p, samples, 48000)
    result = _load_wav(p)
    assert isinstance(result, bytes)
    assert len(result) == 16000 * 2


def test_loader_returns_none_for_no_path():
    loader = ChimeLoader(wake_chime_path=None, search_chime_path=None)
    assert loader.wake_chime is None
    assert loader.search_chime is None


def test_loader_loads_wake_chime(tmp_path):
    p = str(tmp_path / "wake.wav")
    samples = np.arange(16000, dtype=np.int16)
    _make_wav(p, samples, 16000)
    loader = ChimeLoader(wake_chime_path=p, search_chime_path=None)
    assert isinstance(loader.wake_chime, bytes)
    assert loader.search_chime is None


def test_loader_loads_both(tmp_path):
    wp = str(tmp_path / "wake.wav")
    sp = str(tmp_path / "search.wav")
    _make_wav(wp, np.arange(16000, dtype=np.int16), 16000)
    _make_wav(sp, np.arange(8000, dtype=np.int16), 16000)
    loader = ChimeLoader(wake_chime_path=wp, search_chime_path=sp)
    assert isinstance(loader.wake_chime, bytes)
    assert isinstance(loader.search_chime, bytes)
