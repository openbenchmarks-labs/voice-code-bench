from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..http import post_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import wav_audio_bytes

MODAL_ASR_ENDPOINT_DESCRIPTION_BY_SECRET = {
    "MODAL_PARAKEET_ENDPOINT": "Modal ASR endpoint (set MODAL_PARAKEET_ENDPOINT)",
    "MODAL_OMNIASR_ENDPOINT": "Modal ASR endpoint (set MODAL_OMNIASR_ENDPOINT)",
    "MODAL_ORUK_FOURIER_2B_ENDPOINT": "Modal ASR endpoint (set MODAL_ORUK_FOURIER_2B_ENDPOINT)",
}
MODAL_ASR_DEFAULT_SAMPLE_RATE_HZ = 16000
MODAL_ASR_DEFAULT_CHANNELS = 1
MODAL_ASR_DEFAULT_TIMEOUT_SECONDS = 600.0
MODAL_ASR_DEFAULT_MAX_ATTEMPTS = 10


def modal_asr_endpoint(endpoint: str, secret_name: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{secret_name} must be a valid HTTPS URL.")
    return normalized


def modal_asr_payload(audio: bytes, stt_model: SttModelConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "audio_format": "wav",
    }
    language = stt_model.options.get("language")
    if language is None or language == "":
        return payload
    if isinstance(language, bool) or not isinstance(language, str):
        raise ValueError("Modal ASR language must be a string.")
    payload["language"] = language
    return payload


def modal_asr_transcript_from_response(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError("Modal ASR response did not include a non-empty transcript string.")
    transcript = response.get("transcript")
    if not isinstance(transcript, str) or not transcript:
        raise ValueError("Modal ASR response did not include a non-empty transcript string.")
    return transcript


def transcribe(
    audio_path: Path,
    stt_model: SttModelConfig,
    endpoint: str,
    proxy_token_id: str,
    proxy_token_secret: str,
    endpoint_secret_name: str,
) -> TranscriptionResult:
    if stt_model.mode != SttMode.BATCH:
        raise ValueError("Modal ASR transport only supports batch transcription.")
    sample_rate = int(stt_model.options.get("sample_rate", MODAL_ASR_DEFAULT_SAMPLE_RATE_HZ))
    channels = int(stt_model.options.get("channels", MODAL_ASR_DEFAULT_CHANNELS))
    audio = wav_audio_bytes(audio_path, sample_rate=sample_rate, channels=channels)
    response = post_json(
        modal_asr_endpoint(endpoint, endpoint_secret_name),
        modal_asr_payload(audio, stt_model),
        {
            "Modal-Key": proxy_token_id,
            "Modal-Secret": proxy_token_secret,
        },
        timeout=float(stt_model.options.get("timeout_seconds", MODAL_ASR_DEFAULT_TIMEOUT_SECONDS)),
        max_attempts=int(stt_model.options.get("max_attempts", MODAL_ASR_DEFAULT_MAX_ATTEMPTS)),
    )
    return {"transcript": modal_asr_transcript_from_response(response)}
