from __future__ import annotations

import argparse
from pathlib import Path

from voice_code_bench.commands.common import (
    metadata_path_for_available_audio,
    resolve_concurrency,
    run_parallel_tasks,
    run_timestamp,
    validate_positive,
)
from voice_code_bench.commands.extract_entities import run_extract_entities
from voice_code_bench.commands.score import run_score_entities, run_score_wer
from voice_code_bench.commands.transcribe import run_transcribe
from voice_code_bench.io import default_metadata_path, default_secret_path
from voice_code_bench.models import DEFAULT_VERIFIER_ID, STT_SELECTION_MODES, selected_stt_models
from voice_code_bench.verifier import VERIFIER_CACHE_MODES, verifier_config_for_args


def run_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental: run live STT transcription, verification, and scoring.")
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to runs/full-<UTC timestamp>.")
    parser.add_argument("--stt-model-ids", default=None, help="Comma-delimited model IDs. If supplied, --stt-mode is ignored.")
    parser.add_argument("--stt-mode", choices=STT_SELECTION_MODES, default="all")
    parser.add_argument("--secret-path", type=Path, default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-concurrency", type=int, default=None, help="Defaults to the selected model count.")
    parser.add_argument("--entity-concurrency", type=int, default=4)
    parser.add_argument("--verifier-id", default=DEFAULT_VERIFIER_ID)
    parser.add_argument("--verifier-cache", type=Path, default=None)
    parser.add_argument("--verifier-mode", choices=VERIFIER_CACHE_MODES, default="live")
    parser.add_argument("--evaluation-date", default=None, help="ISO date recorded in output run metadata.")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_benchmark_main(argv: list[str] | None = None) -> None:
    args = run_benchmark_parser().parse_args(argv)
    run_full_benchmark(args)


def run_full_benchmark(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    metadata_path = args.metadata or default_metadata_path(dataset_root)
    output_dir = (args.output_dir or dataset_root / "runs" / f"full-{run_timestamp()}").resolve()
    secret_path = args.secret_path if args.secret_path is not None else default_secret_path(dataset_root)

    if args.limit is not None:
        validate_positive(args.limit, "--limit")
    validate_positive(args.entity_concurrency, "--entity-concurrency")
    verifier_config_for_args(args.verifier_id)
    if args.verifier_mode != "live" and args.verifier_cache is None:
        raise ValueError(f"--verifier-mode={args.verifier_mode} requires --verifier-cache.")

    models = selected_stt_models(args.stt_model_ids, args.stt_mode)
    model_concurrency = resolve_concurrency(args.model_concurrency, len(models), "--model-concurrency")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path_for_available_audio(metadata_path, dataset_root, output_dir)

    def run_one_model_pipeline(model_id: str) -> None:
        run_model_pipeline(args, dataset_root, metadata_path, output_dir, secret_path, model_id)

    run_parallel_tasks(
        [model.id for model in models],
        concurrency=model_concurrency,
        task=run_one_model_pipeline,
        item_label=lambda model_id: model_id,
        finished_message=lambda model_id, _result: f"finished_model_pipeline stt_model={model_id}",
    )

    score_entities_args = argparse.Namespace(
        dataset_root=dataset_root,
        metadata=metadata_path,
        entity_matches_dir=output_dir,
        output_dir=output_dir,
        output_csv=output_dir / "results.csv",
        scores_dir=None,
        limit=args.limit,
    )
    score_wer_args = argparse.Namespace(
        dataset_root=dataset_root,
        metadata=metadata_path,
        transcripts_dir=output_dir,
        output_dir=output_dir,
        output_csv=output_dir / "results.csv",
        scores_dir=None,
        limit=args.limit,
    )
    run_score_entities(score_entities_args)
    run_score_wer(score_wer_args)

    print(f"finished_full_run output_dir={output_dir}", flush=True)
    return output_dir


def run_model_pipeline(
    args: argparse.Namespace,
    dataset_root: Path,
    metadata_path: Path,
    output_dir: Path,
    secret_path: Path | None,
    model_id: str,
) -> None:
    transcribe_args = argparse.Namespace(
        dataset_root=dataset_root,
        metadata=metadata_path,
        output_dir=output_dir,
        stt_model_ids=model_id,
        stt_mode="batch",
        secret_path=secret_path,
        project_id=args.project_id,
        limit=args.limit,
        model_concurrency=1,
        datapoint_concurrency=1,
        evaluation_date=args.evaluation_date,
        resume=args.resume,
    )
    run_transcribe(transcribe_args)

    extract_args = argparse.Namespace(
        dataset_root=dataset_root,
        metadata=metadata_path,
        transcripts_dir=output_dir / model_id,
        output_dir=output_dir,
        verifier_id=args.verifier_id,
        verifier_cache=args.verifier_cache,
        verifier_mode=args.verifier_mode,
        secret_path=secret_path,
        limit=args.limit,
        concurrency=args.entity_concurrency,
        model_concurrency=1,
        resume=args.resume,
    )
    run_extract_entities(extract_args)
