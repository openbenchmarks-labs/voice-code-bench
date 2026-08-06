from __future__ import annotations

from pathlib import Path

from .models import MetadataDatapoint, SttModelConfig, TranscriptRow
from .providers import transcribe


def transcribe_datapoint(
    dataset_root: Path,
    datapoint: MetadataDatapoint,
    stt_model: SttModelConfig,
    secrets: dict[str, str],
    project_id: str | None,
) -> TranscriptRow:
    audio_path = dataset_root / datapoint["audio_path"]
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file for {datapoint['audio_id']}: {audio_path}")
    transcription = transcribe(audio_path, stt_model, secrets, project_id)
    row: TranscriptRow = {
        "audio_id": datapoint["audio_id"],
        "model_transcript": transcription["transcript"],
    }
    if "metadata" in transcription:
        row["metadata"] = transcription["metadata"]
    return row
