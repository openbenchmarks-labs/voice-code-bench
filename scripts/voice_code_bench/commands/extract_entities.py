from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from voice_code_bench.commands.common import (
    apply_limit,
    metadata_order,
    read_complete_rows,
    resolve_concurrency,
    resume_rows_by_audio_id,
    run_parallel_tasks,
    validate_positive,
    write_ordered_json_rows,
)
from voice_code_bench.io import (
    default_metadata_path,
    default_run_output_dir,
    default_secret_path,
    load_secret_file,
    model_artifact_path,
    read_json_run_metadata,
    read_metadata,
    row_data_files_by_model,
)
from voice_code_bench.metrics import aggregate_entity_match_rows
from voice_code_bench.models import DEFAULT_VERIFIER_ID, EntityMatchRow, MetadataDatapoint, TranscriptRow
from voice_code_bench.verifier import (
    VERIFIER_CACHE_MODES,
    VerifierCache,
    VerifierConfig,
    verifier_config_for_args,
    verify_entity_matches,
)


def extract_entities_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2: check which target entities are recoverable from STT transcripts.")
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--transcripts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the parent of --transcripts-dir.")
    parser.add_argument("--verifier-id", default=DEFAULT_VERIFIER_ID)
    parser.add_argument("--verifier-cache", type=Path, default=None)
    parser.add_argument("--verifier-mode", choices=VERIFIER_CACHE_MODES, default="live")
    parser.add_argument("--secret-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model-concurrency", type=int, default=None, help="Defaults to the transcript model count.")
    parser.add_argument("--resume", action="store_true")
    return parser


def extract_entities_main(argv: list[str] | None = None) -> None:
    args = extract_entities_parser().parse_args(argv)
    run_extract_entities(args)


def run_extract_entities(args: argparse.Namespace) -> list[Path]:
    validate_positive(args.concurrency, "--concurrency")
    dataset_root = args.dataset_root.resolve()
    metadata_path = args.metadata or default_metadata_path(dataset_root)
    transcripts_dir = args.transcripts_dir.resolve()
    output_dir = (args.output_dir or default_run_output_dir(transcripts_dir, "transcripts")).resolve()
    secret_path = args.secret_path if args.secret_path is not None else default_secret_path(dataset_root)
    secrets = load_secret_file(secret_path)
    datapoints = apply_limit(read_metadata(metadata_path), args.limit)
    datapoint_by_audio_id = {datapoint["audio_id"]: datapoint for datapoint in datapoints}
    order = metadata_order(datapoints)
    transcript_files = row_data_files_by_model(transcripts_dir, "transcripts")
    if not transcript_files:
        raise RuntimeError(f"No transcript JSON files found in {transcripts_dir}.")
    model_concurrency = resolve_concurrency(args.model_concurrency, len(transcript_files), "--model-concurrency")
    verifier_config = verifier_config_for_args(args.verifier_id)
    verifier_cache = VerifierCache(args.verifier_cache) if args.verifier_cache is not None else None
    if args.verifier_mode != "live" and verifier_cache is None:
        raise ValueError(f"--verifier-mode={args.verifier_mode} requires --verifier-cache.")

    def extract_one_model(item: tuple[str, Path]) -> Path:
        model_id, transcript_path = item
        return extract_model(
            output_dir,
            model_id,
            transcript_path,
            datapoints,
            datapoint_by_audio_id,
            order,
            verifier_config,
            verifier_cache,
            args.verifier_mode,
            secrets,
            args.concurrency,
            args.resume,
        )

    return run_parallel_tasks(
        transcript_files,
        concurrency=model_concurrency,
        task=extract_one_model,
        item_label=lambda item: item[0],
        finished_message=lambda model_id, path: f"finished_entities stt_model={model_id} path={path}",
    )


def extract_model(
    output_dir: Path,
    model_id: str,
    transcript_path: Path,
    datapoints: list[MetadataDatapoint],
    datapoint_by_audio_id: dict[str, MetadataDatapoint],
    order: dict[str, int],
    verifier_config: VerifierConfig,
    verifier_cache: VerifierCache | None,
    verifier_mode: str,
    secrets: dict[str, str],
    concurrency: int,
    resume: bool,
) -> Path:
    transcripts = read_complete_rows(transcript_path, datapoints, artifact_name="transcript")
    run_metadata = entity_match_run_metadata(read_json_run_metadata(transcript_path), verifier_config, verifier_mode)
    path = model_artifact_path(output_dir, model_id, "entity_matches")
    rows_by_audio_id = resume_rows_by_audio_id(path, resume=resume, artifact_name="entity-match")
    pending_transcripts = [row for row in transcripts if row["audio_id"] not in rows_by_audio_id]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                extract_one,
                datapoint_by_audio_id[row["audio_id"]],
                row,
                verifier_config,
                verifier_cache,
                verifier_mode,
                secrets,
            ): row["audio_id"]
            for row in pending_transcripts
        }
        for future in as_completed(futures):
            audio_id = futures[future]
            row = future.result()
            rows_by_audio_id[audio_id] = row
            write_entity_match_rows(path, rows_by_audio_id, order, run_metadata)
            print(f"checked_entities audio_id={audio_id} stt_model={model_id}", flush=True)
    write_entity_match_rows(path, rows_by_audio_id, order, run_metadata)
    return path


def extract_one(
    datapoint: MetadataDatapoint,
    transcript_row: TranscriptRow,
    verifier_config: VerifierConfig,
    verifier_cache: VerifierCache | None,
    verifier_mode: str,
    secrets: dict[str, str],
) -> EntityMatchRow:
    transcript = transcript_row["model_transcript"]
    entity_matches = verify_entity_matches(
        datapoint,
        transcript,
        verifier_config,
        secrets,
        cache=verifier_cache,
        cache_mode=verifier_mode,
    )
    row: EntityMatchRow = {
        "audio_id": datapoint["audio_id"],
        "model_transcript": transcript,
        "entity_matches": entity_matches,
    }
    if "metadata" in transcript_row:
        row["metadata"] = transcript_row["metadata"]
    return row


def entity_match_run_metadata(
    source_run_metadata: dict[str, Any] | None,
    verifier_config: VerifierConfig,
    verifier_mode: str,
) -> dict[str, Any] | None:
    if source_run_metadata is None:
        return None
    run_metadata = dict(source_run_metadata)
    run_metadata["entity_verifier"] = {
        "id": verifier_config.id,
        "provider": verifier_config.provider,
        "model": verifier_config.model,
        "endpoint_or_api": verifier_config.endpoint_or_api,
        "config_digest": verifier_config.digest,
        "cache_mode": verifier_mode,
    }
    return run_metadata


def write_entity_match_rows(
    path: Path,
    rows_by_audio_id: dict[str, EntityMatchRow],
    order: dict[str, int],
    run_metadata: dict[str, Any] | None,
) -> None:
    write_ordered_json_rows(
        path,
        rows_by_audio_id,
        order,
        aggregate=aggregate_entity_match_rows,
        run_metadata=run_metadata,
    )
