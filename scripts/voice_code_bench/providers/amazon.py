from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from ..io import secret_value
from ..models import SttModelConfig, TranscriptionResult
from .audio import run_ffmpeg

AMAZON_SAMPLE_RATE_HZ = 16000
AMAZON_CHUNK_DURATION_SECONDS = 0.1
AMAZON_CHUNK_BYTES = int(AMAZON_SAMPLE_RATE_HZ * 2 * AMAZON_CHUNK_DURATION_SECONDS)


def transcribe(audio_path: Path, stt_model: SttModelConfig, secrets: dict[str, str]) -> TranscriptionResult:
    access_key = secret_value(secrets, "AWS_TRANSCRIBE_ACCESS_KEY_ID")
    secret_key = secret_value(secrets, "AWS_TRANSCRIBE_SECRET_ACCESS_KEY")
    if access_key:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
    if secret_key:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
    with tempfile.TemporaryDirectory(prefix="voice-code-bench-amazon-") as temp_dir:
        pcm_path = Path(temp_dir) / f"{audio_path.stem}.pcm"
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(audio_path),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(AMAZON_SAMPLE_RATE_HZ),
                "-ac",
                "1",
                str(pcm_path),
            ]
        )
        return asyncio.run(_transcribe_pcm(pcm_path, stt_model))


async def _transcribe_pcm(pcm_path: Path, stt_model: SttModelConfig) -> TranscriptionResult:
    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        from amazon_transcribe.handlers import TranscriptResultStreamHandler
    except ImportError as exc:
        raise RuntimeError("Install amazon-transcribe to use amazon_transcribe_streaming.") from exc

    class Handler(TranscriptResultStreamHandler):
        def __init__(self, output_stream: Any) -> None:
            super().__init__(output_stream)
            self.transcripts: list[str] = []

        async def handle_transcript_event(self, transcript_event: Any) -> None:
            results = transcript_event.transcript.results
            for result in results:
                if result.is_partial:
                    continue
                if result.alternatives and result.alternatives[0].transcript:
                    self.transcripts.append(result.alternatives[0].transcript.strip())

    client = TranscribeStreamingClient(region=str(stt_model.options.get("region", "us-east-1")))
    stream = await client.start_stream_transcription(
        language_code=str(stt_model.options.get("language_code", "en-US")),
        media_sample_rate_hz=AMAZON_SAMPLE_RATE_HZ,
        media_encoding="pcm",
    )
    handler = Handler(stream.output_stream)

    async def write_chunks() -> None:
        with pcm_path.open("rb") as handle:
            while True:
                chunk = handle.read(AMAZON_CHUNK_BYTES)
                if not chunk:
                    break
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
                await asyncio.sleep(AMAZON_CHUNK_DURATION_SECONDS)
        await stream.input_stream.end_stream()

    await asyncio.gather(write_chunks(), handler.handle_events())
    return {"transcript": " ".join(handler.transcripts)}
