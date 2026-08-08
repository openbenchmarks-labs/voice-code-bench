from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from ..http import post_multipart_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import audio_duration_seconds, iter_audio_chunks, pcm16_streaming_audio
from .websocket import close_websocket_quietly, create_websocket

OPENAI_REALTIME_SAMPLE_RATE_HZ = 24000
OPENAI_REALTIME_CHANNELS = 1
OPENAI_REALTIME_CHUNK_DURATION_SECONDS = 0.1
OPENAI_REALTIME_DRAIN_TIMEOUT_SECONDS = 120.0
OPENAI_REALTIME_RECV_POLL_TIMEOUT_SECONDS = 0.02
OPENAI_REALTIME_SEND_TIMEOUT_SECONDS = 120.0


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    if stt_model.mode == SttMode.STREAM:
        return _transcribe_streaming(audio_path, stt_model, api_key)
    return _transcribe_batch(audio_path, stt_model, api_key)


def _transcribe_batch(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    fields = {"model": stt_model.model, "response_format": "json"}
    if stt_model.options.get("language"):
        fields["language"] = str(stt_model.options["language"])
    response = post_multipart_json(
        "https://api.openai.com/v1/audio/transcriptions",
        fields,
        "file",
        audio_path,
        {"Authorization": f"Bearer {api_key}"},
        file_content_type="audio/wav",
        timeout=300.0,
    )
    return {"transcript": response["text"]}


def _transcribe_streaming(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    sample_rate = int(stt_model.options.get("sample_rate", OPENAI_REALTIME_SAMPLE_RATE_HZ))
    channels = int(stt_model.options.get("channels", OPENAI_REALTIME_CHANNELS))
    if sample_rate != OPENAI_REALTIME_SAMPLE_RATE_HZ:
        raise ValueError("OpenAI realtime transcription requires 24 kHz PCM input.")
    if channels != OPENAI_REALTIME_CHANNELS:
        raise ValueError("OpenAI realtime transcription requires mono PCM input.")
    chunk_duration = float(stt_model.options.get("chunk_duration_seconds", OPENAI_REALTIME_CHUNK_DURATION_SECONDS))
    chunk_size = int(sample_rate * 2 * channels * chunk_duration)
    pcm_audio, _sample_rate, _channels = pcm16_streaming_audio(
        audio_path,
        stt_model,
        default_sample_rate=OPENAI_REALTIME_SAMPLE_RATE_HZ,
        default_channels=OPENAI_REALTIME_CHANNELS,
    )
    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    ws, _websocket = create_websocket(url, [f"Authorization: Bearer {api_key}"], "OpenAI realtime transcription")
    committed_item_ids: list[str] = []
    previous_item_id_by_item_id: dict[str, str | None] = {}
    completed_by_item_id: dict[str, str] = {}
    delta_by_item_id: dict[str, list[str]] = {}
    try:
        ws.send(json.dumps(_realtime_transcription_session(stt_model)))
        while True:
            payload = json.loads(ws.recv())
            event_type = payload.get("type")
            if event_type == "error":
                raise RuntimeError(f"OpenAI realtime transcription failed: {payload}")
            if event_type == "session.updated":
                break
        receive_timeout = OPENAI_REALTIME_RECV_POLL_TIMEOUT_SECONDS if stt_model.options.get("stream_realtime") else 120.0
        ws.settimeout(receive_timeout)
        for chunk in iter_audio_chunks(pcm_audio, max(1, chunk_size)):
            _send_realtime_event(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
                receive_timeout,
            )
            _drain_realtime_events_for_chunk_interval(
                ws,
                stt_model,
                committed_item_ids,
                previous_item_id_by_item_id,
                completed_by_item_id,
                delta_by_item_id,
            )
        if stt_model.options.get("turn_detection"):
            flush_seconds = float(stt_model.options.get("vad_flush_silence_seconds", 1.0))
            for chunk in iter_audio_chunks(b"\x00" * int(sample_rate * 2 * channels * flush_seconds), max(1, chunk_size)):
                _send_realtime_event(
                    ws,
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    },
                    receive_timeout,
                )
                _drain_realtime_events_for_chunk_interval(
                    ws,
                    stt_model,
                    committed_item_ids,
                    previous_item_id_by_item_id,
                    completed_by_item_id,
                    delta_by_item_id,
                )
        if not stt_model.options.get("turn_detection"):
            _send_realtime_event(ws, {"type": "input_audio_buffer.commit"}, receive_timeout)

        ws.settimeout(1.0)
        deadline = time.monotonic() + _realtime_drain_timeout_seconds(audio_path)
        while time.monotonic() < deadline:
            _drain_realtime_event(
                ws,
                committed_item_ids,
                previous_item_id_by_item_id,
                completed_by_item_id,
                delta_by_item_id,
                break_on_timeout=True,
            )
            if committed_item_ids and len(completed_by_item_id) >= len(committed_item_ids):
                break
    finally:
        close_websocket_quietly(ws)

    item_ids = order_realtime_item_ids(committed_item_ids, previous_item_id_by_item_id) or sorted(
        set(completed_by_item_id) | set(delta_by_item_id)
    )
    transcript_parts = [
        completed_by_item_id.get(item_id) or "".join(delta_by_item_id.get(item_id, []))
        for item_id in item_ids
    ]
    transcript = " ".join(part.strip() for part in transcript_parts if part.strip())
    return {"transcript": transcript}


def _send_realtime_event(ws: Any, payload: dict[str, Any], receive_timeout: float) -> None:
    """Use a normal write timeout without losing the short receive-poll timeout."""
    ws.settimeout(OPENAI_REALTIME_SEND_TIMEOUT_SECONDS)
    try:
        ws.send(json.dumps(payload))
    finally:
        ws.settimeout(receive_timeout)


def _drain_realtime_events_for_chunk_interval(
    ws: Any,
    stt_model: SttModelConfig,
    committed_item_ids: list[str],
    previous_item_id_by_item_id: dict[str, str | None],
    completed_by_item_id: dict[str, str],
    delta_by_item_id: dict[str, list[str]],
) -> None:
    if not stt_model.options.get("stream_realtime"):
        return
    duration = float(stt_model.options.get("chunk_duration_seconds", OPENAI_REALTIME_CHUNK_DURATION_SECONDS))
    if duration <= 0:
        return
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        _drain_realtime_event(
            ws,
            committed_item_ids,
            previous_item_id_by_item_id,
            completed_by_item_id,
            delta_by_item_id,
            break_on_timeout=True,
        )


def _drain_realtime_event(
    ws: Any,
    committed_item_ids: list[str],
    previous_item_id_by_item_id: dict[str, str | None],
    completed_by_item_id: dict[str, str],
    delta_by_item_id: dict[str, list[str]],
    *,
    break_on_timeout: bool,
) -> bool:
    try:
        message = ws.recv()
    except Exception as exc:
        module_name = type(exc).__module__
        class_name = type(exc).__name__
        if class_name in {"WebSocketTimeoutException", "TimeoutError"}:
            return False
        if class_name == "WebSocketConnectionClosedException" or module_name.startswith("websocket."):
            return False
        raise
    if not message:
        return False
    payload = json.loads(message)
    event_type = payload.get("type")
    if event_type == "error":
        raise RuntimeError(f"OpenAI realtime transcription failed: {payload}")
    if event_type == "input_audio_buffer.committed" and payload.get("item_id"):
        item_id = str(payload["item_id"])
        if item_id not in committed_item_ids:
            committed_item_ids.append(item_id)
        previous_item_id = payload.get("previous_item_id")
        previous_item_id_by_item_id[item_id] = str(previous_item_id) if previous_item_id else None
    elif event_type == "conversation.item.input_audio_transcription.delta" and payload.get("item_id"):
        delta_by_item_id.setdefault(str(payload["item_id"]), []).append(str(payload.get("delta") or ""))
    elif event_type == "conversation.item.input_audio_transcription.completed" and payload.get("item_id"):
        completed_by_item_id[str(payload["item_id"])] = str(payload.get("transcript") or "")
    return True


def order_realtime_item_ids(item_ids: list[str], previous_item_id_by_item_id: dict[str, str | None]) -> list[str]:
    if not item_ids:
        return []
    remaining = set(item_ids)
    ordered: list[str] = []
    previous = next(
        (item_id for item_id in item_ids if previous_item_id_by_item_id.get(item_id) not in remaining),
        item_ids[0],
    )
    while previous in remaining:
        ordered.append(previous)
        remaining.remove(previous)
        previous = next(
            (item_id for item_id in item_ids if item_id in remaining and previous_item_id_by_item_id.get(item_id) == previous),
            "",
        )
    ordered.extend(item_id for item_id in item_ids if item_id in remaining)
    return ordered


def _realtime_transcription_session(stt_model: SttModelConfig) -> dict[str, Any]:
    transcription: dict[str, Any] = {"model": stt_model.model}
    if stt_model.options.get("language"):
        transcription["language"] = str(stt_model.options["language"])
    if stt_model.options.get("prompt"):
        transcription["prompt"] = str(stt_model.options["prompt"])
    audio_input: dict[str, Any] = {
        "format": {
            "type": "audio/pcm",
            "rate": int(stt_model.options.get("sample_rate", OPENAI_REALTIME_SAMPLE_RATE_HZ)),
        },
        "transcription": transcription,
    }
    if "turn_detection" in stt_model.options:
        audio_input["turn_detection"] = stt_model.options["turn_detection"]
    if stt_model.options.get("input_audio_noise_reduction"):
        audio_input["noise_reduction"] = stt_model.options["input_audio_noise_reduction"]
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {"input": audio_input},
        },
    }


def _realtime_drain_timeout_seconds(audio_path: Path) -> float:
    return max(
        OPENAI_REALTIME_DRAIN_TIMEOUT_SECONDS,
        audio_duration_seconds(audio_path) + OPENAI_REALTIME_DRAIN_TIMEOUT_SECONDS,
    )
