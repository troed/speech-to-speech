"""
Start a local websocket-mode server and test it end-to-end.
This isolates from the user's server setup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from queue import Queue, Empty
from threading import Event, Thread

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_local")

sys.path.insert(0, "src")

# Import needed modules
from speech_to_speech.s2s_pipeline import (
    build_pipeline,
    initialize_queues_and_events,
    parse_arguments,
    prepare_all_args,
)
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.wake_word_arguments import WakeWordHandlerArguments
from speech_to_speech.utils.thread_manager import ThreadManager


def run_test():
    # Parse CLI args or use defaults
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument("--ws-port", type=int, default=18766)
    args, remaining = parser.parse_known_args()

    # Replace sys.argv for HfArgumentParser
    old_argv = sys.argv
    sys.argv = [
        "s2s_pipeline.py",
        "--mode", "websocket",
        "--stt", "parakeet-tdt",
        "--llm_backend", "responses-api",
        "--tts", "qwen3",
        "--ws_port", str(args.ws_port),
        "--log_level", "debug",
        "--enable_live_transcription", "false",
    ]

    try:
        queues_and_events = initialize_queues_and_events()
        module_kwargs, socket_receiver_kwargs, socket_sender_kwargs, websocket_streamer_kwargs, vad_handler_kwargs, whisper_stt_handler_kwargs, faster_whisper_stt_handler_kwargs, paraformer_stt_handler_kwargs, mlx_audio_whisper_stt_handler_kwargs, parakeet_tdt_stt_handler_kwargs, language_model_handler_kwargs, responses_api_language_model_handler_kwargs, chat_tts_handler_kwargs, facebook_mms_tts_handler_kwargs, pocket_tts_handler_kwargs, kokoro_tts_handler_kwargs, qwen3_tts_handler_kwargs = parse_arguments()

        module_kwargs: ModuleArguments
        websocket_streamer_kwargs: WebSocketStreamerArguments  # type: ignore

        # Override model settings for minimal test
        module_kwargs.stt = "parakeet-tdt"
        module_kwargs.llm_backend = "responses-api"
        module_kwargs.tts = "qwen3"

        logger.info("Starting pipeline with mode=%s, port=%d", module_kwargs.mode, args.ws_port)

        thread_manager = build_pipeline(
            module_kwargs=module_kwargs,
            socket_receiver_kwargs=socket_receiver_kwargs,
            socket_sender_kwargs=socket_sender_kwargs,
            websocket_streamer_kwargs=websocket_streamer_kwargs,
            vad_handler_kwargs=vad_handler_kwargs,
            whisper_stt_handler_kwargs=whisper_stt_handler_kwargs,
            faster_whisper_stt_handler_kwargs=faster_whisper_stt_handler_kwargs,
            paraformer_stt_handler_kwargs=paraformer_stt_handler_kwargs,
            mlx_audio_whisper_stt_handler_kwargs=mlx_audio_whisper_stt_handler_kwargs,
            parakeet_tdt_stt_handler_kwargs=parakeet_tdt_stt_handler_kwargs,
            language_model_handler_kwargs=language_model_handler_kwargs,
            responses_api_language_model_handler_kwargs=responses_api_language_model_handler_kwargs,
            chat_tts_handler_kwargs=chat_tts_handler_kwargs,
            facebook_mms_tts_handler_kwargs=facebook_mms_tts_handler_kwargs,
            pocket_tts_handler_kwargs=pocket_tts_handler_kwargs,
            kokoro_tts_handler_kwargs=kokoro_tts_handler_kwargs,
            qwen3_tts_handler_kwargs=qwen3_tts_handler_kwargs,
            queues_and_events=queues_and_events,
        )

        thread_manager.start()
        logger.info("Pipeline started, listening on ws://127.0.0.1:%d", args.ws_port)

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            thread_manager.stop()
            thread_manager.join()

    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    run_test()
