"""Pre-download all models needed by the pipeline.

Run once before first use so the pipeline never needs to reach out to
HuggingFace / GitHub / NLTK at startup::

    uv run python scripts/download_models.py
"""

from __future__ import annotations

import logging
import os

# Temporarily allow network so downloads succeed.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("download_models")


def _dl(description: str, fn, *args, **kwargs) -> None:
    logger.info("Downloading %s …", description)
    fn(*args, **kwargs)
    logger.info("  ✓ %s", description)


def download_nltk() -> None:
    import nltk

    for resource in ("tokenizers/punkt_tab", "taggers/averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(resource)
            logger.info("  ✓ NLTK %s (already cached)", resource)
        except (LookupError, OSError):
            _dl(f"NLTK {resource}", nltk.download, resource)


def download_silero_vad() -> None:
    import torch

    hub_dir = torch.hub.get_dir()
    cached = (
        any("snakers4_silero-vad" in d for d in os.listdir(hub_dir))
        if os.path.isdir(hub_dir)
        else False
    )
    if cached:
        logger.info("  ✓ Silero VAD (already cached)")
    else:
        _dl("Silero VAD", torch.hub.load, "snakers4/silero-vad", "silero_vad", trust_repo=True)


def download_parakeet_tdt() -> None:
    try:
        from nano_parakeet import from_pretrained

        model = "nvidia/parakeet-tdt-0.6b-v3"
        _dl(f"Parakeet TDT ({model})", from_pretrained, model_name=model, device="cpu")
    except ImportError:
        logger.warning("nano-parakeet not installed — skipping Parakeet TDT download")


def download_qwen3_tts() -> None:
    try:
        from faster_qwen3_tts import FasterQwen3TTS

        model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        _dl(f"Qwen3-TTS ({model})", FasterQwen3TTS.from_pretrained, model, device="cpu")
    except ImportError:
        logger.warning("faster-qwen3-tts not installed — skipping Qwen3-TTS download")


def download_all() -> None:
    download_nltk()
    download_silero_vad()
    download_parakeet_tdt()
    download_qwen3_tts()
    logger.info("\nAll models downloaded. The pipeline can now run offline.")


if __name__ == "__main__":
    download_all()
