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
        default=30.0,
        metadata={
            "help": "Seconds of silence before the handler goes back to sleep after wake word activation."
        },
    )
    preroll_ms: int = field(
        default=1000,
        metadata={
            "help": "Milliseconds of audio to retain before the detected wake word and forward on activation."
        },
    )
