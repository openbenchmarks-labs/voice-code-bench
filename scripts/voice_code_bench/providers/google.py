from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..http import post_json
from ..io import secret_value
from ..models import SttMode, SttModelConfig, TranscriptionResult
from .audio import audio_duration_seconds, iter_audio_chunks, pcm16_audio_bytes, require_executable, run_ffmpeg

GOOGLE_STREAM_SAMPLE_RATE_HZ = 16000
GOOGLE_STREAM_CHANNELS = 1
GOOGLE_STREAM_CHUNK_DURATION_SECONDS = 0.5
GOOGLE_STREAM_PCM_CHUNK_BYTES = int(
    GOOGLE_STREAM_SAMPLE_RATE_HZ
    * 2
    * GOOGLE_STREAM_CHANNELS
    * GOOGLE_STREAM_CHUNK_DURATION_SECONDS
)
GOOGLE_STREAM_MIN_TIMEOUT_SECONDS = 300.0
GOOGLE_STREAM_TIMEOUT_PADDING_SECONDS = 180.0


def transcribe(
    audio_path: Path,
    stt_model: SttModelConfig,
    secrets: dict[str, str],
    project_id: str | None,
) -> TranscriptionResult:
    if stt_model.mode == SttMode.STREAM:
        return _transcribe_streaming(audio_path, stt_model, secrets, project_id)
    return _transcribe_batch(audio_path, stt_model, secrets, project_id)


def _transcribe_batch(
    audio_path: Path,
    stt_model: SttModelConfig,
    secrets: dict[str, str],
    project_id: str | None,
) -> TranscriptionResult:
    resolved_project_id = _resolve_project_id(secrets, project_id)
    token = _access_token()
    region = str(stt_model.options.get("region", "us"))
    language_code = str(stt_model.options.get("language_code", "en-US"))
    with tempfile.TemporaryDirectory(prefix="voice-code-bench-google-") as temp_dir:
        chunk_pattern = str(Path(temp_dir) / "chunk_%03d.flac")
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(audio_path),
                "-f",
                "segment",
                "-segment_time",
                "55",
                "-ar",
                "48000",
                "-ac",
                "1",
                chunk_pattern,
            ]
        )
        transcripts: list[str] = []
        for chunk_path in sorted(Path(temp_dir).glob("chunk_*.flac")):
            response = post_json(
                (
                    f"https://{region}-speech.googleapis.com/v2/projects/{resolved_project_id}"
                    f"/locations/{region}/recognizers/_:recognize"
                ),
                {
                    "config": {
                        "autoDecodingConfig": {},
                        "languageCodes": [language_code],
                        "model": stt_model.model,
                    },
                    "content": base64.b64encode(chunk_path.read_bytes()).decode("ascii"),
                },
                {"Authorization": f"Bearer {token}"},
                timeout=300.0,
            )
            for result in response.get("results", []):
                alternatives = result.get("alternatives") or []
                if alternatives and alternatives[0].get("transcript"):
                    transcripts.append(alternatives[0]["transcript"].strip())
    return {"transcript": " ".join(item for item in transcripts if item)}


def _transcribe_streaming(
    audio_path: Path,
    stt_model: SttModelConfig,
    secrets: dict[str, str],
    project_id: str | None,
) -> TranscriptionResult:
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise RuntimeError("Install google-cloud-speech to use Google Cloud streaming STT.") from exc

    resolved_project_id = _resolve_project_id(secrets, project_id)
    region = str(stt_model.options.get("region", "us"))
    language_code = str(stt_model.options.get("language_code", "en-US"))
    client = SpeechClient(
        credentials=Credentials(token=_access_token()),
        client_options=ClientOptions(api_endpoint=f"{region}-speech.googleapis.com"),
    )
    audio_content = pcm16_audio_bytes(
        audio_path,
        sample_rate=GOOGLE_STREAM_SAMPLE_RATE_HZ,
        channels=GOOGLE_STREAM_CHANNELS,
    )
    audio_chunks = list(iter_audio_chunks(audio_content, GOOGLE_STREAM_PCM_CHUNK_BYTES))
    audio_requests = _stream_audio_requests(
        audio_chunks,
        chunk_duration_seconds=GOOGLE_STREAM_CHUNK_DURATION_SECONDS,
        cloud_speech_types=cloud_speech_types,
    )
    recognition_config = cloud_speech_types.RecognitionConfig(
        explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
            encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=GOOGLE_STREAM_SAMPLE_RATE_HZ,
            audio_channel_count=GOOGLE_STREAM_CHANNELS,
        ),
        language_codes=[language_code],
        model=stt_model.model,
    )
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(config=recognition_config)
    config_request = cloud_speech_types.StreamingRecognizeRequest(
        recognizer=f"projects/{resolved_project_id}/locations/{region}/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests() -> Any:
        yield config_request
        yield from audio_requests

    transcripts: list[str] = []
    for response in client.streaming_recognize(requests=requests(), timeout=_stream_timeout_seconds(audio_path)):
        for result in response.results:
            alternatives = result.alternatives
            if alternatives and alternatives[0].transcript:
                transcripts.append(alternatives[0].transcript.strip())
    return {"transcript": " ".join(item for item in transcripts if item)}


def _stream_audio_requests(
    audio_chunks: list[bytes],
    *,
    chunk_duration_seconds: float,
    cloud_speech_types: Any,
) -> Any:
    for index, chunk in enumerate(audio_chunks):
        yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)
        if index < len(audio_chunks) - 1 and chunk_duration_seconds > 0:
            time.sleep(chunk_duration_seconds)


def _resolve_project_id(secrets: dict[str, str], project_id: str | None) -> str:
    resolved_project_id = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or secret_value(secrets, "GOOGLE_CLOUD_PROJECT")
        or secret_value(secrets, "GCLOUD_PROJECT")
        or secret_value(secrets, "GCP_PROJECT")
    )
    if not resolved_project_id:
        raise RuntimeError("Missing Google Cloud project ID. Pass --project-id or set GOOGLE_CLOUD_PROJECT.")
    return resolved_project_id


def _access_token() -> str:
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        if credentials.token:
            return credentials.token
    except Exception:
        pass
    require_executable("gcloud", "Install the gcloud CLI or configure Google application-default credentials.")
    commands = [["gcloud", "auth", "application-default", "print-access-token"]]
    auth_list = subprocess.run(
        ["gcloud", "auth", "list", "--format=value(account)"],
        text=True,
        capture_output=True,
    )
    if auth_list.returncode == 0:
        for account in [line.strip() for line in auth_list.stdout.splitlines() if line.strip()]:
            commands.append(["gcloud", "auth", "print-access-token", f"--account={account}"])
    last_error = ""
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
        last_error = completed.stderr.strip()
    raise RuntimeError(f"Unable to obtain a Google access token. {last_error}")


def _stream_timeout_seconds(audio_path: Path) -> float:
    return max(
        GOOGLE_STREAM_MIN_TIMEOUT_SECONDS,
        audio_duration_seconds(audio_path) + GOOGLE_STREAM_TIMEOUT_PADDING_SECONDS,
    )
