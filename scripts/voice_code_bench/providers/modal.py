from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..http import post_json
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import wav_audio_bytes

MODAL_ENDPOINT_DESCRIPTION = "Modal managed endpoint (set MODAL_INKLING_ENDPOINT)"
MODAL_DEFAULT_SAMPLE_RATE_HZ = 16000
MODAL_DEFAULT_CHANNELS = 1
MODAL_DEFAULT_PROMPT = "Transcribe the following speech to text."
MODAL_DEFAULT_MAX_TOKENS = 4096
MODAL_DEFAULT_REASONING_EFFORT = "max"
MODAL_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "max"}
MODAL_DEFAULT_TIMEOUT_SECONDS = 600.0
MODAL_DEFAULT_MAX_ATTEMPTS = 10


def transcribe(
    audio_path: Path,
    stt_model: SttModelConfig,
    endpoint: str,
    proxy_token_id: str,
    proxy_token_secret: str,
) -> TranscriptionResult:
    if stt_model.mode != SttMode.BATCH:
        raise ValueError("Modal Inkling is only supported through batch chat completions.")
    sample_rate = int(stt_model.options.get("sample_rate", MODAL_DEFAULT_SAMPLE_RATE_HZ))
    channels = int(stt_model.options.get("channels", MODAL_DEFAULT_CHANNELS))
    audio = wav_audio_bytes(audio_path, sample_rate=sample_rate, channels=channels)
    payload = modal_transcription_payload(audio, stt_model)
    response = post_json(
        modal_chat_completions_endpoint(endpoint),
        payload,
        {
            "Modal-Key": proxy_token_id,
            "Modal-Secret": proxy_token_secret,
        },
        timeout=float(stt_model.options.get("timeout_seconds", MODAL_DEFAULT_TIMEOUT_SECONDS)),
        max_attempts=int(stt_model.options.get("max_attempts", MODAL_DEFAULT_MAX_ATTEMPTS)),
        retryable_empty_error_status_codes={400},
    )
    return {"transcript": modal_transcript_from_response(response)}


def modal_chat_completions_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("MODAL_INKLING_ENDPOINT must be a valid HTTPS URL.")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def modal_transcription_payload(
    audio: bytes,
    stt_model: SttModelConfig,
) -> dict[str, Any]:
    prompt = stt_model.options.get("prompt", MODAL_DEFAULT_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Modal Inkling transcription prompt must be a non-empty string.")
    max_tokens = stt_model.options.get("max_tokens", MODAL_DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("Modal Inkling max_tokens must be a positive integer.")
    reasoning_effort = stt_model.options.get("reasoning_effort", MODAL_DEFAULT_REASONING_EFFORT)
    if not isinstance(reasoning_effort, str) or reasoning_effort not in MODAL_REASONING_EFFORTS:
        allowed = ", ".join(sorted(MODAL_REASONING_EFFORTS))
        raise ValueError(f"Modal Inkling reasoning_effort must be one of: {allowed}.")
    encoded_audio = base64.b64encode(audio).decode("ascii")
    return {
        "model": stt_model.model,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{encoded_audio}"},
                    },
                ],
            }
        ],
    }


def modal_transcript_from_response(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Modal Inkling response did not include message content.") from exc
    if not isinstance(content, str):
        raise ValueError("Modal Inkling response message content must be a string.")
    return content
