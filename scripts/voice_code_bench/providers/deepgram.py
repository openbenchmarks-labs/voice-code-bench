from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

from ..http import request_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import STREAM_CHANNELS, STREAM_CHUNK_BYTES, STREAM_SAMPLE_RATE_HZ, iter_audio_chunks, maybe_sleep_between_stream_chunks, pcm16_streaming_audio
from .websocket import close_websocket_quietly, create_websocket, iter_websocket_json_until_closed


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    if stt_model.mode == SttMode.STREAM:
        return _transcribe_streaming(audio_path, stt_model, api_key)
    return _transcribe_batch(audio_path, stt_model, api_key)


def _transcribe_batch(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    query = urllib.parse.urlencode({"model": stt_model.model})
    response = request_json(
        "POST",
        f"https://api.deepgram.com/v1/listen?{query}",
        audio_path.read_bytes(),
        {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        timeout=300.0,
    )
    transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    return {"transcript": transcript}


def _transcribe_streaming(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    pcm_audio, sample_rate, channels = pcm16_streaming_audio(
        audio_path,
        stt_model,
        default_sample_rate=STREAM_SAMPLE_RATE_HZ,
        default_channels=STREAM_CHANNELS,
    )
    query = {
        "model": stt_model.model,
        "encoding": str(stt_model.options.get("encoding", "linear16")),
        "sample_rate": str(sample_rate),
        "channels": str(channels),
    }
    if stt_model.options.get("language"):
        query["language"] = str(stt_model.options["language"])
    url = f"wss://api.deepgram.com/v1/listen?{urllib.parse.urlencode(query)}"
    ws, websocket = create_websocket(url, [f"Authorization: Token {api_key}"], "Deepgram streaming STT")
    final_transcripts: list[str] = []
    interim_transcripts: list[str] = []
    try:
        for chunk in iter_audio_chunks(pcm_audio, STREAM_CHUNK_BYTES):
            ws.send_binary(chunk)
            maybe_sleep_between_stream_chunks(stt_model)
        ws.send(json.dumps({"type": "CloseStream"}))
        for payload in iter_websocket_json_until_closed(ws, websocket):
            if payload.get("type") != "Results":
                continue
            alternatives = payload.get("channel", {}).get("alternatives") or []
            if not alternatives:
                continue
            transcript = str(alternatives[0].get("transcript") or "").strip()
            if not transcript:
                continue
            if payload.get("is_final"):
                final_transcripts.append(transcript)
            else:
                interim_transcripts.append(transcript)
    finally:
        close_websocket_quietly(ws)
    transcripts = final_transcripts or interim_transcripts
    return {"transcript": " ".join(transcripts)}
