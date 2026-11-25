from __future__ import annotations

from pathlib import Path

from ..http import post_multipart_json
from ..models import SttModelConfig, TranscriptionResult

ORUK_TRANSCRIPTIONS_ENDPOINT = "https://speech-api.oruk.ai/v1/audio/transcriptions"


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    fields = {
        "model": stt_model.model,
        "response_format": "json",
    }
    if stt_model.options.get("language"):
        fields["language"] = str(stt_model.options["language"])
    response = post_multipart_json(
        ORUK_TRANSCRIPTIONS_ENDPOINT,
        fields,
        "file",
        audio_path,
        {"Authorization": f"Bearer {api_key}"},
        file_content_type="audio/wav",
        timeout=300.0,
    )
    return {"transcript": response["text"]}
