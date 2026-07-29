from __future__ import annotations

import asyncio
import sys
import time
import types
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np
import pytest

from speech_to_speech.client import WakeWordDetector


class _MockModel:
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores or [0.0]
        self._idx = 0
        self._reset_calls = 0

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        score = self._scores[min(self._idx, len(self._scores) - 1)]
        self._idx += 1
        return {"test_model": score}

    def reset(self) -> None:
        self._reset_calls += 1
        self._idx = 0


def _silence_chunk() -> bytes:
    return np.zeros(512, dtype=np.int16).tobytes()


def _make_detector(threshold: float = 0.5, scores: list[float] | None = None) -> WakeWordDetector:
    return WakeWordDetector(
        model_path="fake.onnx",
        threshold=threshold,
        wake_chime_bytes=None,
    )


def _patch_model(monkeypatch, model: _MockModel) -> _MockModel:
    mock_openwakeword = types.ModuleType("openwakeword")
    mock_model_mod = types.ModuleType("openwakeword.model")
    mock_model_mod.Model = lambda **kw: model
    mock_openwakeword.model = mock_model_mod

    monkeypatch.setitem(sys.modules, "openwakeword", mock_openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mock_model_mod)

    return model


class TestWakeWordDetector:
    def test_does_not_trigger_on_silence(self, monkeypatch):
        model = _MockModel(scores=[0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        for _ in range(50):
            mic_queue.put(_silence_chunk())

        time.sleep(0.3)
        assert not wake_event.is_set(), "detector should not trigger on silence"

    def test_resets_model_after_detection(self, monkeypatch):
        model = _MockModel(scores=[1.0, 0.0, 0.0, 0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        mic_queue.put(_silence_chunk())
        time.sleep(0.2)
        assert wake_event.is_set(), "detector should trigger on high-score chunk"
        assert model._reset_calls == 1, "model.reset() should be called after detection"

    def test_no_false_trigger_after_reset_and_silence(self, monkeypatch):
        model = _MockModel(scores=[1.0, 0.0, 0.0, 0.0, 0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        mic_queue.put(_silence_chunk())
        time.sleep(0.2)
        assert wake_event.is_set()

        wake_event.clear()
        time.sleep(0.1)

        for _ in range(30):
            mic_queue.put(_silence_chunk())

        time.sleep(0.3)
        assert not wake_event.is_set(), "detector should not re-trigger on silence after reset"

    def test_detector_sleeps_when_wake_active(self, monkeypatch):
        model = _MockModel(scores=[1.0, 1.0, 1.0, 1.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        wake_event.set()

        for _ in range(20):
            mic_queue.put(_silence_chunk())

        time.sleep(0.3)

        drained: list[bytes] = []
        while not mic_queue.empty():
            try:
                drained.append(mic_queue.get_nowait())
            except Empty:
                break

        assert len(drained) >= 15, (
            f"detector should not consume mic_queue while wake is active, "
            f"expected >=15 drained, got {len(drained)}"
        )

    def test_detector_wakes_after_wake_cleared(self, monkeypatch):
        model = _MockModel(scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        wake_event.set()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        time.sleep(0.2)

        for _ in range(10):
            mic_queue.put(_silence_chunk())

        time.sleep(0.2)
        still_in_queue = 0
        while not mic_queue.empty():
            try:
                mic_queue.get_nowait()
                still_in_queue += 1
            except Empty:
                break
        assert still_in_queue >= 8, (
            f"detector should not consume while wake is set, items remaining: {still_in_queue}"
        )

        wake_event.clear()

        time.sleep(0.5)

        drained = 0
        while not mic_queue.empty():
            try:
                mic_queue.get_nowait()
                drained += 1
            except Empty:
                break
        assert drained <= 1, (
            f"detector should drain mic_queue including settling chunks, "
            f"items remaining: {drained}"
        )

    def test_resets_model_on_wake_to_sleep_transition(self, monkeypatch):
        model = _MockModel(scores=[1.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        mic_queue.put(_silence_chunk())
        time.sleep(0.2)
        assert wake_event.is_set()
        assert model._reset_calls == 1

        wake_event.clear()

        time.sleep(0.3)
        assert model._reset_calls >= 2, (
            f"model.reset() should be called on wake→sleep transition, "
            f"got {model._reset_calls} calls"
        )


class TestSendReceiveCoordination:
    def test_response_timeout_clears_wake(self, monkeypatch):
        model = _MockModel(scores=[0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        speaker_queue: Queue[bytes] = Queue()
        stop_event = Event()
        wake_event = Event()
        wake_event.set()

        response_active = True
        last_recv_audio = time.monotonic() - 5.0

        async def _run():
            nonlocal response_active
            while not stop_event.is_set():
                if not wake_event.is_set():
                    await asyncio.sleep(0.1)
                    continue
                if response_active and time.monotonic() - last_recv_audio > 1.5 and speaker_queue.empty():
                    response_active = False
                    wake_event.clear()
                    while not mic_queue.empty():
                        try:
                            mic_queue.get_nowait()
                        except Empty:
                            break
                    continue

                try:
                    await asyncio.to_thread(mic_queue.get, True, 0.1)
                except Empty:
                    continue

        async def _test():
            task = asyncio.create_task(_run())
            await asyncio.sleep(0.3)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_test())
        assert not wake_event.is_set(), "wake_event should be cleared after response timeout"
        assert not response_active

    def test_response_timeout_waits_for_speaker_drain(self, monkeypatch):
        model = _MockModel(scores=[0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        speaker_queue: Queue[bytes] = Queue()
        stop_event = Event()
        wake_event = Event()
        wake_event.set()

        speaker_queue.put(b"\x00" * 1024)

        response_active = True
        last_recv_audio = time.monotonic() - 5.0

        async def _run():
            nonlocal response_active
            while not stop_event.is_set():
                if not wake_event.is_set():
                    await asyncio.sleep(0.1)
                    continue
                if response_active and time.monotonic() - last_recv_audio > 1.5 and speaker_queue.empty():
                    response_active = False
                    wake_event.clear()
                    while not mic_queue.empty():
                        try:
                            mic_queue.get_nowait()
                        except Empty:
                            break
                    continue

                try:
                    await asyncio.to_thread(mic_queue.get, True, 0.1)
                except Empty:
                    continue

        async def _test():
            task = asyncio.create_task(_run())
            await asyncio.sleep(0.3)
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_test())
        assert wake_event.is_set(), "wake_event should still be set when speaker_queue is not empty"

    def test_wake_re_detected_does_not_loop_with_silence(self, monkeypatch):
        model = _MockModel(scores=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _patch_model(monkeypatch, model)

        mic_queue: Queue[bytes] = Queue()
        speaker_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = _make_detector(threshold=0.5)
        detector.start(mic_queue, wake_event)

        mic_queue.put(_silence_chunk())
        time.sleep(0.2)
        assert wake_event.is_set(), "first detection should trigger"
        assert model._reset_calls == 1

        wake_event.clear()
        time.sleep(0.1)

        for _ in range(30):
            mic_queue.put(_silence_chunk())

        time.sleep(0.4)
        assert not wake_event.is_set(), (
            "detector should NOT re-trigger on silence after reset"
        )


class TestRealModelSilence:
    """Reproduction: feed real wake word model with silence and check for false triggers."""

    MODEL_PATH = "computer.onnx"

    @pytest.fixture(autouse=True)
    def _require_openwakeword(self):
        pytest.importorskip("openwakeword")

    def test_real_model_does_not_trigger_on_silence(self):
        from openwakeword.model import Model

        model = Model(wakeword_model_paths=[self.MODEL_PATH])
        chunk = _silence_chunk()
        max_score = 0.0
        for _ in range(200):
            audio = np.frombuffer(chunk, dtype=np.int16)
            prediction = model.predict(audio)
            if prediction:
                score = max(prediction.values())
                if score > max_score:
                    max_score = score

        threshold = 0.5
        assert max_score < threshold, (
            f"real model should not trigger on silence: max_score={max_score:.4f}, threshold={threshold}"
        )

    def test_real_model_false_trigger_rate_on_silence(self):
        from openwakeword.model import Model

        false_count = 0
        total_chunks = 500
        threshold = 0.5

        for trial in range(3):
            model = Model(wakeword_model_paths=[self.MODEL_PATH])
            for _ in range(total_chunks):
                audio = np.frombuffer(_silence_chunk(), dtype=np.int16)
                prediction = model.predict(audio)
                if prediction and max(prediction.values()) >= threshold:
                    false_count += 1

        rate = false_count / (3 * total_chunks) * 100
        assert rate == 0.0, (
            f"false trigger rate on silence: {rate:.2f}% ({false_count}/{3*total_chunks})"
        )

    def test_real_model_state_after_sleep_wake_cycle(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])

        def _detector_loop():
            was_active = False
            for _ in range(300):
                if wake_event.is_set():
                    was_active = True
                    time.sleep(0.1)
                    continue
                if was_active:
                    was_active = False
                    model.reset()
                    for _2 in range(5):
                        try:
                            mic_queue.get_nowait()
                        except Empty:
                            break
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction and max(prediction.values()) >= threshold:
                    if not wake_event.is_set():
                        wake_event.set()

        Thread(target=_detector_loop, daemon=True).start()

        wake_event.set()
        time.sleep(0.3)

        wake_event.clear()
        time.sleep(0.3)

        for _ in range(100):
            mic_queue.put(_silence_chunk())

        time.sleep(1.0)
        assert not wake_event.is_set(), (
            "detector should NOT re-trigger on silence after sleep→wake cycle with real model"
        )

    def test_real_model_does_not_trigger_on_speech_like_noise(self):
        from openwakeword.model import Model

        model = Model(wakeword_model_paths=[self.MODEL_PATH])
        threshold = 0.5

        for _ in range(300):
            noise = (np.random.randn(512) * 2000).astype(np.int16)
            prediction = model.predict(noise)
            if prediction and max(prediction.values()) >= threshold:
                top = max(prediction, key=prediction.get)
                pytest.fail(f"model triggered on noise: {top}={prediction[top]:.4f}")

    def test_real_model_does_not_trigger_on_tones(self):
        from openwakeword.model import Model

        model = Model(wakeword_model_paths=[self.MODEL_PATH])
        threshold = 0.5
        t = np.arange(512) / 16000.0

        for freq in [300, 500, 800, 1000, 2000, 3000]:
            tone = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
            for _ in range(50):
                prediction = model.predict(tone)
                if prediction and max(prediction.values()) >= threshold:
                    top = max(prediction, key=prediction.get)
                    pytest.fail(f"model triggered on {freq}Hz tone: {top}={prediction[top]:.4f}")

    def test_real_model_no_false_trigger_after_sleep_without_reset(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])

        def _old_behavior_loop():
            for _ in range(300):
                if wake_event.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction and max(prediction.values()) >= threshold:
                    if not wake_event.is_set():
                        wake_event.set()

        Thread(target=_old_behavior_loop, daemon=True).start()

        wake_event.set()
        time.sleep(0.5)

        wake_event.clear()
        time.sleep(0.3)

        for _ in range(100):
            mic_queue.put(_silence_chunk())

        time.sleep(1.0)
        assert not wake_event.is_set(), (
            "old behavior (no reset on wake→sleep) should not re-trigger on silence either"
        )

    def test_real_model_scores_after_long_sleep_reveal_state(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])
        scores: list[float] = []

        def _collector():
            for _ in range(400):
                if wake_event.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction:
                    scores.append(max(prediction.values()))

        Thread(target=_collector, daemon=True).start()

        noise = (np.random.randn(512) * 4000).astype(np.int16).tobytes()
        for _ in range(50):
            mic_queue.put(noise)
        time.sleep(0.5)

        wake_event.set()
        time.sleep(2.0)

        wake_event.clear()
        time.sleep(0.3)

        for _ in range(200):
            mic_queue.put(_silence_chunk())

        time.sleep(1.0)

        if scores:
            max_after = max(scores)
            assert max_after < threshold, (
                f"model scores after long sleep should stay below threshold: "
                f"max={max_after:.4f}, n_samples={len(scores)}"
            )

    def test_production_scenario_old_behavior_long_sleep(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])

        triggered = False

        def _old_behavior():
            nonlocal triggered
            for _ in range(2000):
                if wake_event.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction and max(prediction.values()) >= threshold:
                    if not wake_event.is_set():
                        triggered = True
                        wake_event.set()

        Thread(target=_old_behavior, daemon=True).start()

        for _ in range(100):
            mic_queue.put(_silence_chunk())
        time.sleep(0.5)

        mic_queue.put(_silence_chunk())
        time.sleep(0.3)
        assert wake_event.is_set() or not triggered

        wake_event.clear()
        time.sleep(15.0)

        wake_event.set()
        time.sleep(15.0)

        wake_event.clear()
        time.sleep(0.5)

        for _ in range(500):
            mic_queue.put(_silence_chunk())

        time.sleep(3.0)
        assert not triggered, (
            "OLD behavior (no reset on wake→sleep) should NOT trigger after 15s sleep"
        )

    def test_production_scenario_noise_to_15s_sleep_to_silence(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])

        triggered = False

        def _detector():
            nonlocal triggered
            was_active = False
            for _ in range(3000):
                if wake_event.is_set():
                    was_active = True
                    time.sleep(0.1)
                    continue
                if was_active:
                    was_active = False
                    model.reset()
                    for _2 in range(5):
                        try:
                            mic_queue.get_nowait()
                        except Empty:
                            break
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction:
                    score = max(prediction.values())
                    if score >= threshold and not wake_event.is_set():
                        triggered = True
                        wake_event.set()

        Thread(target=_detector, daemon=True).start()

        noise = (np.random.randn(512) * 2000).astype(np.int16).tobytes()
        for _ in range(200):
            mic_queue.put(noise)
        time.sleep(2.0)

        wake_event.set()
        time.sleep(15.0)

        wake_event.clear()
        time.sleep(0.5)

        for _ in range(500):
            mic_queue.put(_silence_chunk())

        time.sleep(3.0)
        assert not triggered, (
            "detector should NOT trigger after 15s sleep regardless of pre-sleep audio"
        )

    def test_production_scenario_30s_sleep_then_silence(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])

        triggered = False

        def _detector():
            nonlocal triggered
            for _ in range(5000):
                if wake_event.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction:
                    score = max(prediction.values())
                    if score >= threshold and not wake_event.is_set():
                        triggered = True
                        wake_event.set()

        Thread(target=_detector, daemon=True).start()

        noise = (np.random.randn(512) * 3000).astype(np.int16).tobytes()
        for _ in range(400):
            mic_queue.put(noise)
        time.sleep(2.0)

        wake_event.set()
        time.sleep(30.0)

        wake_event.clear()
        time.sleep(0.5)

        for _ in range(800):
            mic_queue.put(_silence_chunk())

        time.sleep(4.0)
        assert not triggered, (
            "OLD behavior should NOT trigger after 30s sleep on silence"
        )

    def test_instrumented_full_loop_reveals_trigger_source(self):
        from openwakeword.model import Model

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()
        threshold = 0.5

        model = Model(wakeword_model_paths=[self.MODEL_PATH])
        original_predict = model.predict
        predictions_log: list[float] = []

        def _wrapped_predict(x):
            result = original_predict(x)
            if result:
                score = max(result.values())
                predictions_log.append(score)
            else:
                predictions_log.append(0.0)
            return result

        model.predict = _wrapped_predict  # type: ignore[method-assign]

        triggered = False
        trigger_scores: list[float] = []

        def _detector():
            nonlocal triggered
            was_active = False
            for _ in range(3000):
                if wake_event.is_set():
                    was_active = True
                    time.sleep(0.1)
                    continue
                if was_active:
                    was_active = False
                    model.reset()
                    for _2 in range(5):
                        try:
                            mic_queue.get_nowait()
                        except Empty:
                            break
                try:
                    chunk_bytes = mic_queue.get(timeout=0.1)
                except Empty:
                    continue
                audio = np.frombuffer(chunk_bytes, dtype=np.int16)
                prediction = model.predict(audio)
                if prediction:
                    score = max(prediction.values())
                    if score >= threshold and not wake_event.is_set():
                        triggered = True
                        trigger_scores.append(score)
                        wake_event.set()

        Thread(target=_detector, daemon=True).start()

        for _ in range(100):
            mic_queue.put(_silence_chunk())
        time.sleep(1.0)

        wake_event.set()
        time.sleep(5.0)

        wake_event.clear()
        time.sleep(0.5)

        for _ in range(500):
            mic_queue.put(_silence_chunk())

        time.sleep(2.0)

        max_after_wake = max(predictions_log[-50:]) if len(predictions_log) >= 50 else 0
        assert not triggered, (
            f"detector triggered on silence: trigger_scores={trigger_scores}, "
            f"last 10 predictions={predictions_log[-10:] if predictions_log else 'none'}, "
            f"max post-wake={max_after_wake:.6f}"
        )


def _load_wav_16k_mono(path: str) -> bytes:
    import wave

    with wave.open(path, "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = raw.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.float64)
    target_len = int(len(audio) * 16000 / w.getframerate())
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, target_len), np.arange(len(audio)), audio
    ).astype(np.int16)
    return resampled.tobytes()


def _feed_wav_to_queue(wav_bytes: bytes, q: Queue):
    chunk_size = 1024  # 512 samples * 2 bytes
    for offset in range(0, len(wav_bytes), chunk_size):
        chunk = wav_bytes[offset : offset + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        q.put(chunk)
        time.sleep(0.032)  # simulate real-time audio: 32ms per 512-sample chunk


class TestProductionReproduction:
    """Full-scenario reproduction using real voice recordings."""

    MODEL_PATH = "computer.onnx"
    WAKE_WORD_WAV = "test_wakeword.wav"
    QUESTION_WAV = "test_question.wav"

    @pytest.fixture(autouse=True)
    def _require(self):
        pytest.importorskip("openwakeword")
        from openwakeword.model import Model

    def _make_audio(self, path: str) -> bytes:
        return _load_wav_16k_mono(path)

    def test_wakeword_triggers_detector(self):
        from openwakeword.model import Model

        wake_audio = self._make_audio(self.WAKE_WORD_WAV)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = WakeWordDetector(
            model_path=self.MODEL_PATH,
            threshold=0.5,
            wake_chime_bytes=None,
        )
        detector.start(mic_queue, wake_event)

        Thread(target=lambda: _feed_wav_to_queue(wake_audio, mic_queue), daemon=True).start()
        time.sleep(2.5)

        assert wake_event.is_set(), "wake word audio MUST trigger the detector"

    def test_question_audio_does_not_trigger_detector(self):
        from openwakeword.model import Model

        question_audio = self._make_audio(self.QUESTION_WAV)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = WakeWordDetector(
            model_path=self.MODEL_PATH,
            threshold=0.5,
            wake_chime_bytes=None,
        )
        detector.start(mic_queue, wake_event)

        Thread(target=lambda: _feed_wav_to_queue(question_audio, mic_queue), daemon=True).start()
        time.sleep(3.0)

        assert not wake_event.is_set(), "question audio MUST NOT trigger the detector"

    def test_detector_does_not_re_trigger_after_wake_then_question(self):
        from openwakeword.model import Model

        wake_audio = self._make_audio(self.WAKE_WORD_WAV)
        question_audio = self._make_audio(self.QUESTION_WAV)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = WakeWordDetector(
            model_path=self.MODEL_PATH,
            threshold=0.5,
            wake_chime_bytes=None,
        )
        detector.start(mic_queue, wake_event)

        Thread(target=lambda: _feed_wav_to_queue(wake_audio, mic_queue), daemon=True).start()
        time.sleep(2.5)
        assert wake_event.is_set(), "first detection must trigger"

        Thread(target=lambda: _feed_wav_to_queue(question_audio, mic_queue), daemon=True).start()
        time.sleep(1.0)

        wake_event.clear()
        while True:
            try:
                mic_queue.get_nowait()
            except Empty:
                break

        time.sleep(0.3)

        assert not wake_event.is_set(), (
            "detector MUST NOT re-trigger after reliable queue drain"
        )

    def test_detector_does_not_re_trigger_after_wake_then_silence(self):
        from openwakeword.model import Model

        wake_audio = self._make_audio(self.WAKE_WORD_WAV)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = WakeWordDetector(
            model_path=self.MODEL_PATH,
            threshold=0.5,
            wake_chime_bytes=None,
        )
        detector.start(mic_queue, wake_event)

        Thread(target=lambda: _feed_wav_to_queue(wake_audio, mic_queue), daemon=True).start()
        time.sleep(2.5)
        assert wake_event.is_set()

        wake_event.clear()
        time.sleep(0.5)

        for _ in range(200):
            mic_queue.put(_silence_chunk())

        time.sleep(2.0)
        assert not wake_event.is_set(), "detector MUST NOT re-trigger on silence after wake->sleep->wake"

    def test_detector_correctly_retriggers_on_second_wake_word(self):
        from openwakeword.model import Model

        wake_audio = self._make_audio(self.WAKE_WORD_WAV)

        mic_queue: Queue[bytes] = Queue()
        wake_event = Event()

        detector = WakeWordDetector(
            model_path=self.MODEL_PATH,
            threshold=0.5,
            wake_chime_bytes=None,
        )
        detector.start(mic_queue, wake_event)

        Thread(target=lambda: _feed_wav_to_queue(wake_audio, mic_queue), daemon=True).start()
        time.sleep(2.5)
        assert wake_event.is_set()

        wake_event.clear()
        time.sleep(0.5)

        for _ in range(300):
            mic_queue.put(_silence_chunk())
        time.sleep(1.0)
        assert not wake_event.is_set(), "should not trigger on silence"

        wake_event.clear()
        Thread(target=lambda: _feed_wav_to_queue(wake_audio, mic_queue), daemon=True).start()
        time.sleep(2.5)

        assert wake_event.is_set(), "second wake word should trigger detector"

