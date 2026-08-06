from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..http import post_json, request_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import STREAM_CHANNELS, STREAM_CHUNK_BYTES, STREAM_SAMPLE_RATE_HZ, iter_audio_chunks, maybe_sleep_between_stream_chunks, pcm16_streaming_audio
from .websocket import close_websocket_quietly, create_websocket, iter_websocket_json_until_closed

ASSEMBLYAI_POLL_INTERVAL_SECONDS = 3
ASSEMBLYAI_TIMEOUT_SECONDS = 60 * 60


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    if stt_model.mode == SttMode.STREAM:
        return _transcribe_streaming(audio_path, stt_model, api_key)
    return _transcribe_batch(audio_path, stt_model, api_key)


def _transcribe_batch(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    headers = {"authorization": api_key}
    upload = request_json("POST", "https://api.assemblyai.com/v2/upload", audio_path.read_bytes(), headers, timeout=300.0)
    request_body: dict[str, Any] = {"audio_url": upload["upload_url"], "speech_models": [stt_model.model]}
    if stt_model.options.get("language_code"):
        request_body["language_code"] = stt_model.options["language_code"]
    job = post_json("https://api.assemblyai.com/v2/transcript", request_body, headers, timeout=120.0)
    transcript_id = job["id"]
    deadline = time.monotonic() + ASSEMBLYAI_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = request_json("GET", f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        status = result["status"]
        if status == "completed":
            return {"transcript": result["text"]}
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {result.get('error')}")
        time.sleep(ASSEMBLYAI_POLL_INTERVAL_SECONDS)
    raise TimeoutError("AssemblyAI transcription timed out.")


def _transcribe_streaming(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    pcm_audio, sample_rate, _channels = pcm16_streaming_audio(
        audio_path,
        stt_model,
        default_sample_rate=STREAM_SAMPLE_RATE_HZ,
        default_channels=STREAM_CHANNELS,
    )
    query = {
        "sample_rate": str(sample_rate),
        "speech_model": str(stt_model.options.get("speech_model", stt_model.model)),
        "encoding": str(stt_model.options.get("encoding", "pcm_s16le")),
    }
    if "format_turns" in stt_model.options:
        query["format_turns"] = _json_bool(stt_model.options["format_turns"])
    url = f"wss://streaming.assemblyai.com/v3/ws?{urllib.parse.urlencode(query)}"
    ws, websocket = create_websocket(url, [f"Authorization: {api_key}"], "AssemblyAI streaming STT")
    turns: dict[int, str] = {}
    fallback_transcripts: list[str] = []
    try:
        for chunk in iter_audio_chunks(pcm_audio, STREAM_CHUNK_BYTES):
            ws.send_binary(chunk)
            maybe_sleep_between_stream_chunks(stt_model)
        ws.send(json.dumps({"type": "Terminate"}))
        for payload in iter_websocket_json_until_closed(ws, websocket):
            transcript = str(payload.get("transcript") or "").strip()
            if not transcript:
                continue
            turn_order = payload.get("turn_order")
            if isinstance(turn_order, int):
                turns[turn_order] = transcript
            else:
                fallback_transcripts.append(transcript)
    finally:
        close_websocket_quietly(ws)
    transcripts = [turns[index] for index in sorted(turns)] if turns else fallback_transcripts
    return {"transcript": " ".join(transcripts)}


def _json_bool(value: Any) -> str:
    return "true" if bool(value) else "false"
