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
GOOGLE_STREAM_MIN_TIMEOUT_SECONDS = 180.0
GOOGLE_STREAM_TIMEOUT_PADDING_SECONDS = 90.0
GOOGLE_STREAM_MAX_METADATA_BYTES = 64 * 1024
GOOGLE_STREAM_RECOVERY_SEGMENT_SECONDS = 45.0
GOOGLE_STREAM_FLUSH_SILENCE_SECONDS = 1.0


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
        from google.api_core.exceptions import DeadlineExceeded
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.services.speech.transports.grpc import SpeechGrpcTransport
        from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise RuntimeError("Install google-cloud-speech to use Google Cloud streaming STT.") from exc

    resolved_project_id = _resolve_project_id(secrets, project_id)
    region = str(stt_model.options.get("region", "us"))
    language_code = str(stt_model.options.get("language_code", "en-US"))
    credentials = Credentials(token=_access_token())

    def new_client() -> Any:
        channel = SpeechGrpcTransport.create_channel(
            f"{region}-speech.googleapis.com",
            credentials=credentials,
            options=google_stream_channel_options(),
        )
        return SpeechClient(transport=SpeechGrpcTransport(channel=channel))

    client = new_client()
    audio_content = pcm16_audio_bytes(
        audio_path,
        sample_rate=GOOGLE_STREAM_SAMPLE_RATE_HZ,
        channels=GOOGLE_STREAM_CHANNELS,
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
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech_types.StreamingRecognitionFeatures(interim_results=True),
    )
    recognizer = f"projects/{resolved_project_id}/locations/{region}/recognizers/_"
    try:
        transcripts = _recognize_streaming_content(
            client,
            audio_content,
            recognizer=recognizer,
            streaming_config=streaming_config,
            cloud_speech_types=cloud_speech_types,
            timeout=_stream_timeout_seconds(audio_path),
        )
    except DeadlineExceeded:
        client.transport.close()
        transcripts = []
        for segment in _split_pcm_audio(
            audio_content,
            sample_rate=GOOGLE_STREAM_SAMPLE_RATE_HZ,
            channels=GOOGLE_STREAM_CHANNELS,
            segment_seconds=GOOGLE_STREAM_RECOVERY_SEGMENT_SECONDS,
        ):
            recovery_client = new_client()
            try:
                transcripts.extend(
                    _recognize_streaming_content(
                        recovery_client,
                        segment,
                        recognizer=recognizer,
                        streaming_config=streaming_config,
                        cloud_speech_types=cloud_speech_types,
                        timeout=GOOGLE_STREAM_MIN_TIMEOUT_SECONDS,
                    )
                )
            finally:
                recovery_client.transport.close()
    else:
        client.transport.close()
    return {"transcript": " ".join(item for item in transcripts if item)}


def _recognize_streaming_content(
    client: Any,
    audio_content: bytes,
    *,
    recognizer: str,
    streaming_config: Any,
    cloud_speech_types: Any,
    timeout: float,
) -> list[str]:
    audio_content = _append_pcm_silence(
        audio_content,
        sample_rate=GOOGLE_STREAM_SAMPLE_RATE_HZ,
        channels=GOOGLE_STREAM_CHANNELS,
        silence_seconds=GOOGLE_STREAM_FLUSH_SILENCE_SECONDS,
    )
    audio_chunks = list(iter_audio_chunks(audio_content, GOOGLE_STREAM_PCM_CHUNK_BYTES))

    def requests() -> Any:
        yield cloud_speech_types.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=streaming_config,
        )
        yield from _stream_audio_requests(
            audio_chunks,
            chunk_duration_seconds=GOOGLE_STREAM_CHUNK_DURATION_SECONDS,
            cloud_speech_types=cloud_speech_types,
        )

    responses = client.streaming_recognize(requests=requests(), timeout=timeout)
    return _streaming_response_transcripts(responses)


def _append_pcm_silence(
    audio_content: bytes,
    *,
    sample_rate: int,
    channels: int,
    silence_seconds: float,
) -> bytes:
    silence_frames = max(0, int(sample_rate * silence_seconds))
    return audio_content + bytes(silence_frames * channels * 2)


def _streaming_response_transcripts(responses: Any) -> list[str]:
    transcripts: list[str] = []
    pending_interim = ""
    for response in responses:
        for result in response.results:
            alternatives = result.alternatives
            if not alternatives or not alternatives[0].transcript:
                continue
            transcript = alternatives[0].transcript.strip()
            if result.is_final:
                transcripts.append(transcript)
                pending_interim = ""
            else:
                pending_interim = transcript
    if pending_interim:
        transcripts.append(pending_interim)
    return transcripts


def _split_pcm_audio(
    audio_content: bytes,
    *,
    sample_rate: int,
    channels: int,
    segment_seconds: float,
) -> list[bytes]:
    bytes_per_frame = 2 * channels
    segment_frames = max(1, int(sample_rate * segment_seconds))
    segment_bytes = segment_frames * bytes_per_frame
    return [
        audio_content[offset : offset + segment_bytes]
        for offset in range(0, len(audio_content), segment_bytes)
    ]


def google_stream_channel_options() -> list[tuple[str, int]]:
    return [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        ("grpc.max_metadata_size", GOOGLE_STREAM_MAX_METADATA_BYTES),
    ]


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
