from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from voice_code_bench.commands.common import (
    apply_limit,
    metadata_order,
    parse_evaluation_date,
    resolve_concurrency,
    resume_rows_by_audio_id,
    run_parallel_tasks,
    run_timestamp,
    validate_positive,
    write_ordered_json_rows,
)
from voice_code_bench.io import (
    default_metadata_path,
    default_secret_path,
    load_secret_file,
    model_artifact_path,
    read_metadata,
)
from voice_code_bench.metrics import aggregate_transcript_rows
from voice_code_bench.models import (
    STT_SELECTION_MODES,
    MetadataDatapoint,
    SttModelConfig,
    TranscriptRow,
    selected_stt_models,
    stt_run_metadata,
)
from voice_code_bench.runner import transcribe_datapoint


def transcribe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental: transcribe benchmark audio with live STT providers.")
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to runs/transcribe-<UTC timestamp>.")
    parser.add_argument("--stt-model-ids", default=None, help="Comma-delimited model IDs. If supplied, --stt-mode is ignored.")
    parser.add_argument("--stt-mode", choices=STT_SELECTION_MODES, default="batch")
    parser.add_argument("--secret-path", type=Path, default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-concurrency", type=int, default=None, help="Defaults to the selected model count.")
    parser.add_argument("--datapoint-concurrency", type=int, default=1)
    parser.add_argument("--evaluation-date", default=None, help="ISO date recorded in output run metadata. Defaults to today in UTC.")
    parser.add_argument("--resume", action="store_true")
    return parser


def transcribe_main(argv: list[str] | None = None) -> None:
    args = transcribe_parser().parse_args(argv)
    run_transcribe(args)


def run_transcribe(args: argparse.Namespace) -> list[Path]:
    dataset_root = args.dataset_root.resolve()
    metadata_path = args.metadata or default_metadata_path(dataset_root)
    output_dir = (args.output_dir or dataset_root / "runs" / f"transcribe-{run_timestamp()}").resolve()
    secret_path = args.secret_path if args.secret_path is not None else default_secret_path(dataset_root)
    secrets = load_secret_file(secret_path)
    evaluation_date = parse_evaluation_date(args.evaluation_date)
    datapoints = apply_limit(read_metadata(metadata_path), args.limit)
    models = selected_stt_models(args.stt_model_ids, args.stt_mode)
    validate_positive(args.datapoint_concurrency, "--datapoint-concurrency")
    model_concurrency = resolve_concurrency(args.model_concurrency, len(models), "--model-concurrency")
    order = metadata_order(datapoints)

    def transcribe_one_model(model: SttModelConfig) -> Path:
        return transcribe_model(
            dataset_root,
            output_dir,
            datapoints,
            model,
            secrets,
            args.project_id,
            order,
            evaluation_date,
            args.resume,
            args.datapoint_concurrency,
        )

    return run_parallel_tasks(
        models,
        concurrency=model_concurrency,
        task=transcribe_one_model,
        item_label=lambda model: model.id,
        finished_message=lambda model_id, path: f"finished stt_model={model_id} path={path}",
    )


def transcribe_model(
    dataset_root: Path,
    output_dir: Path,
    datapoints: list[MetadataDatapoint],
    model: SttModelConfig,
    secrets: dict[str, str],
    project_id: str | None,
    order: dict[str, int],
    evaluation_date: str,
    resume: bool,
    datapoint_concurrency: int,
) -> Path:
    path = model_artifact_path(output_dir, model.id, "transcripts")
    rows_by_audio_id = resume_rows_by_audio_id(path, resume=resume, artifact_name="transcript")
    pending_datapoints = [datapoint for datapoint in datapoints if datapoint["audio_id"] not in rows_by_audio_id]
    run_metadata = stt_run_metadata(model, evaluation_date, secrets)
    if datapoint_concurrency == 1:
        for datapoint in pending_datapoints:
            try:
                row = transcribe_datapoint(dataset_root, datapoint, model, secrets, project_id)
            except Exception as exc:
                if not _is_deadline_exceeded(exc):
                    raise
                row = _deadline_exceeded_row(datapoint["audio_id"])
                print(
                    f"skipped audio_id={datapoint['audio_id']} stt_model={model.id} reason=deadline_exceeded",
                    flush=True,
                )
            rows_by_audio_id[row["audio_id"]] = row
            write_transcript_rows(path, rows_by_audio_id, order, run_metadata)
            if not row.get("metadata", {}).get("skipped"):
                print(f"transcribed audio_id={datapoint['audio_id']} stt_model={model.id}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=datapoint_concurrency) as executor:
            futures = {
                executor.submit(transcribe_datapoint, dataset_root, datapoint, model, secrets, project_id): datapoint[
                    "audio_id"
                ]
                for datapoint in pending_datapoints
            }
            for future in as_completed(futures):
                audio_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    if not _is_deadline_exceeded(exc):
                        raise
                    row = _deadline_exceeded_row(audio_id)
                    print(
                        f"skipped audio_id={audio_id} stt_model={model.id} reason=deadline_exceeded",
                        flush=True,
                    )
                rows_by_audio_id[row["audio_id"]] = row
                write_transcript_rows(path, rows_by_audio_id, order, run_metadata)
                if not row.get("metadata", {}).get("skipped"):
                    print(f"transcribed audio_id={audio_id} stt_model={model.id}", flush=True)
    write_transcript_rows(path, rows_by_audio_id, order, run_metadata)
    return path


def _is_deadline_exceeded(exc: Exception) -> bool:
    return type(exc).__name__ == "DeadlineExceeded"


def _deadline_exceeded_row(audio_id: str) -> TranscriptRow:
    return {
        "audio_id": audio_id,
        "model_transcript": "",
        "metadata": {
            "skipped": True,
            "skip_reason": "deadline_exceeded",
        },
    }


def write_transcript_rows(
    path: Path,
    rows_by_audio_id: dict[str, TranscriptRow],
    order: dict[str, int],
    run_metadata: dict[str, Any],
) -> None:
    write_ordered_json_rows(
        path,
        rows_by_audio_id,
        order,
        aggregate=aggregate_transcript_rows,
        run_metadata=run_metadata,
    )
