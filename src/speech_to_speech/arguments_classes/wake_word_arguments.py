from dataclasses import dataclass, field


@dataclass
class WakeWordHandlerArguments:
    model_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to openWakeWord .tflite model file. When not set, wake word detection is disabled (current behavior)."
        },
    )
    threshold: float = field(
        default=0.5,
        metadata={
            "help": "Detection confidence threshold (0-1). Lower values increase sensitivity but may increase false positives."
        },
    )
    activation_timeout_s: float = field(
        default=10.0,
        metadata={
            "help": "Seconds of inactivity after the last response finishes before requiring a new wake word activation."
        },
    )
    preroll_ms: int = field(
        default=1000,
        metadata={
            "help": "Milliseconds of audio to retain before the detected wake word and forward on activation."
        },
    )
    wake_chime: str | None = field(
        default=None,
        metadata={
            "help": "Path to WAV file to play when wake word is detected (signals user can speak)."
        },
    )
    search_chime: str | None = field(
        default=None,
        metadata={
            "help": "Path to WAV file to play after a server-side tool/search returns (signals answer incoming)."
        },
    )
