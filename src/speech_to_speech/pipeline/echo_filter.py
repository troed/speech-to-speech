from __future__ import annotations

import re
import threading


class EchoFilter:
    def __init__(self, threshold: float = 0.9, max_history: int = 3) -> None:
        self._lock = threading.Lock()
        self._threshold = threshold
        self._max_history = max_history
        self._texts: list[str] = []

    def record(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._texts.append(text)
            if len(self._texts) > self._max_history:
                self._texts.pop(0)

    def is_echo(self, candidate: str) -> bool:
        candidate = candidate.strip()
        if not candidate:
            return False
        with self._lock:
            for text in self._texts:
                if self._texts_similar(candidate, text):
                    return True
        return False

    def _texts_similar(self, a: str, b: str) -> bool:
        wa = set(re.findall(r"\w+", a.lower()))
        wb = set(re.findall(r"\w+", b.lower()))
        if not wa or not wb:
            return False
        smaller = min(len(wa), len(wb))
        overlap = len(wa & wb)
        return overlap / smaller >= self._threshold
