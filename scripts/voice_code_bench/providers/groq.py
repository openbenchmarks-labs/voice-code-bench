from __future__ import annotations

from pathlib import Path

from ..http import post_multipart_json
from ..models import SttModelConfig, TranscriptionResult


def transcribe(audio_path: Path, stt_model: SttModelConfig, api_key: str) -> TranscriptionResult:
    fields = {
        "model": "whisper-large-v3" if stt_model.model == "large-v3" else stt_model.model,
        "response_format": "json",
        "temperature": "0",
    }
    if stt_model.options.get("language"):
        fields["language"] = str(stt_model.options["language"])
    response = post_multipart_json(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        fields,
        "file",
        audio_path,
        {"Authorization": f"Bearer {api_key}"},
        file_content_type="audio/wav",
        timeout=300.0,
    )
    return {"transcript": response["text"]}
