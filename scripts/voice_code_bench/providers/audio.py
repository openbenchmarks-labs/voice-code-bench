from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from ..models import SttModelConfig

STREAM_SAMPLE_RATE_HZ = 16000
STREAM_CHANNELS = 1
STREAM_CHUNK_DURATION_SECONDS = 0.1
STREAM_CHUNK_BYTES = int(STREAM_SAMPLE_RATE_HZ * 2 * STREAM_CHUNK_DURATION_SECONDS)


def run_ffmpeg(args: list[str]) -> None:
    require_executable("ffmpeg", "Install ffmpeg to run provider audio conversion.")
    try:
        subprocess.run(["ffmpeg", *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
        detail = f": {stderr[-500:]}" if stderr else ""
        raise RuntimeError(f"ffmpeg audio conversion failed{detail}") from exc


def require_executable(name: str, message: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(message)


def pcm16_audio_bytes(audio_path: Path, *, sample_rate: int, channels: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="voice-code-bench-pcm-") as temp_dir:
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
                str(sample_rate),
                "-ac",
                str(channels),
                str(pcm_path),
            ]
        )
        return pcm_path.read_bytes()


def wav_audio_bytes(audio_path: Path, *, sample_rate: int, channels: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="voice-code-bench-wav-") as temp_dir:
        wav_path = Path(temp_dir) / f"{audio_path.stem}.wav"
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(audio_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(wav_path),
            ]
        )
        return wav_path.read_bytes()


def pcm16_streaming_audio(
    audio_path: Path,
    stt_model: SttModelConfig,
    *,
    default_sample_rate: int = STREAM_SAMPLE_RATE_HZ,
    default_channels: int = STREAM_CHANNELS,
) -> tuple[bytes, int, int]:
    sample_rate = int(stt_model.options.get("sample_rate", default_sample_rate))
    channels = int(stt_model.options.get("channels", default_channels))
    return pcm16_audio_bytes(audio_path, sample_rate=sample_rate, channels=channels), sample_rate, channels


def iter_audio_chunks(audio: bytes, chunk_size: int) -> Any:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if chunk:
            yield chunk


def audio_duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate > 0:
                return handle.getnframes() / frame_rate
    except (EOFError, wave.Error):
        pass
    return 0.0


def maybe_sleep_between_stream_chunks(stt_model: SttModelConfig) -> None:
    if not stt_model.options.get("stream_realtime"):
        return
    duration = float(stt_model.options.get("chunk_duration_seconds", STREAM_CHUNK_DURATION_SECONDS))
    if duration > 0:
        time.sleep(duration)
