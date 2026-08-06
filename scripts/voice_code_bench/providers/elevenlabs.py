from __future__ import annotations

import base64
import json
import urllib.parse
from pathlib import Path
from typing import Any

from ..http import post_multipart_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import STREAM_CHANNELS, STREAM_CHUNK_BYTES, STREAM_SAMPLE_RATE_HZ, iter_audio_chunks, maybe_sleep_between_stream_chunks, pcm16_streaming_audio
from .websocket import close_websocket_quietly, create_websocket, iter_websocket_json_until_closed


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    if stt_model.mode == SttMode.STREAM:
        return _transcribe_streaming(audio_path, stt_model, api_key)
    return _transcribe_batch(audio_path, stt_model, api_key)


def _transcribe_batch(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    fields = _form_fields({"model_id": stt_model.model}, stt_model.options)
    response = post_multipart_json(
        "https://api.elevenlabs.io/v1/speech-to-text",
        fields,
        "file",
        audio_path,
        {"xi-api-key": api_key},
        file_content_type="audio/wav",
        timeout=300.0,
    )
    return {"transcript": response["text"]}


def _transcribe_streaming(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    pcm_audio, sample_rate, _channels = pcm16_streaming_audio(
        audio_path,
        stt_model,
        default_sample_rate=STREAM_SAMPLE_RATE_HZ,
        default_channels=STREAM_CHANNELS,
    )
    query = {
        "model_id": stt_model.model,
        "audio_format": str(stt_model.options.get("audio_format", f"pcm_{sample_rate}")),
        "commit_strategy": str(stt_model.options.get("commit_strategy", "manual")),
    }
    for key in [
        "language_code",
        "include_timestamps",
        "include_language_detection",
        "enable_logging",
        "vad_silence_threshold_secs",
        "vad_threshold",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
    ]:
        if key in stt_model.options:
            query[key] = _query_value(stt_model.options[key])
    url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{urllib.parse.urlencode(query)}"
    ws, websocket = create_websocket(url, [f"xi-api-key: {api_key}"], "ElevenLabs streaming STT")
    idle_timeout_seconds = float(stt_model.options.get("idle_timeout_seconds", 10.0))
    committed_transcripts: list[str] = []
    partial_transcripts: list[str] = []
    try:
        for chunk in iter_audio_chunks(pcm_audio, STREAM_CHUNK_BYTES):
            ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                        "sample_rate": sample_rate,
                    }
                )
            )
            maybe_sleep_between_stream_chunks(stt_model)
        ws.send(
            json.dumps(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "commit": True,
                    "sample_rate": sample_rate,
                }
            )
        )
        for payload in iter_websocket_json_until_closed(ws, websocket):
            message_type = payload.get("message_type")
            transcript = str(payload.get("text") or "").strip()
            if _is_error_message(message_type):
                raise RuntimeError(f"ElevenLabs streaming STT failed: {payload}")
            if not transcript:
                continue
            if message_type in {"committed_transcript", "committed_transcript_with_timestamps"}:
                if not committed_transcripts or committed_transcripts[-1] != transcript:
                    committed_transcripts.append(transcript)
                ws.settimeout(idle_timeout_seconds)
            elif message_type == "partial_transcript":
                partial_transcripts.append(transcript)
    finally:
        close_websocket_quietly(ws)
    transcripts = committed_transcripts or partial_transcripts
    return {"transcript": " ".join(transcripts)}


def _json_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return _json_bool(value)
    return str(value)


def _form_fields(base_fields: dict[str, str], options: dict[str, Any]) -> dict[str, str]:
    fields = dict(base_fields)
    for key in [
        "language_code",
        "tag_audio_events",
        "num_speakers",
        "timestamps_granularity",
        "diarize",
        "file_format",
        "temperature",
        "seed",
        "use_multi_channel",
        "entity_detection",
        "no_verbatim",
    ]:
        if key not in options:
            continue
        value = options[key]
        if value is None:
            continue
        fields[key] = _query_value(value)
    return fields


def _is_error_message(message_type: Any) -> bool:
    if not isinstance(message_type, str):
        return False
    return message_type == "error" or message_type.endswith("_error")
