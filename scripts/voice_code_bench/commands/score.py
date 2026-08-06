from __future__ import annotations

import argparse
from pathlib import Path

from voice_code_bench.commands.common import apply_limit, read_complete_rows, sort_model_artifacts
from voice_code_bench.io import (
    default_metadata_path,
    default_run_output_dir,
    model_artifact_path,
    read_json_run_metadata,
    read_metadata,
    row_data_files_by_model,
    write_json_rows,
)
from voice_code_bench.metrics import (
    aggregate_entity_score_rows,
    aggregate_entity_scores,
    aggregate_wer_score_rows,
    aggregate_wer_scores,
    score_entity_capture,
    score_wer,
    write_entity_results_csv,
    write_wer_results_csv,
)


def score_entities_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 3: score entity capture from entity-match JSON files.")
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--entity-matches-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the parent of --entity-matches-dir.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Defaults to results.csv in --output-dir.")
    parser.add_argument("--scores-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def score_entities_main(argv: list[str] | None = None) -> None:
    args = score_entities_parser().parse_args(argv)
    run_score_entities(args)


def run_score_entities(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    metadata_path = args.metadata or default_metadata_path(dataset_root)
    entity_matches_dir = args.entity_matches_dir.resolve()
    output_dir = (args.output_dir or default_run_output_dir(entity_matches_dir, "entity_matches")).resolve()
    output_csv = args.output_csv or output_dir / "results.csv"
    scores_dir = args.scores_dir
    datapoints = apply_limit(read_metadata(metadata_path), args.limit)
    entity_match_files = row_data_files_by_model(entity_matches_dir, "entity_matches")
    if not entity_match_files:
        raise RuntimeError(f"No entity-match JSON files found in {entity_matches_dir}.")
    sort_model_artifacts(entity_match_files)

    all_scores = []
    model_ids = []
    rows = []
    for model_id, path in entity_match_files:
        entity_rows = read_complete_rows(path, datapoints, artifact_name="entity-match")
        rows_by_audio_id = {row["audio_id"]: row for row in entity_rows}
        scores = [score_entity_capture(datapoint, model_id, rows_by_audio_id[datapoint["audio_id"]]) for datapoint in datapoints]
        score_path = scores_dir / f"{model_id}.json" if scores_dir else model_artifact_path(output_dir, model_id, "entity_scores")
        write_json_rows(score_path, scores, aggregate=aggregate_entity_score_rows(scores), run_metadata=read_json_run_metadata(path))
        all_scores.extend(scores)
        model_ids.append(model_id)
        rows = aggregate_entity_scores(all_scores, model_ids, len(datapoints))
        write_entity_results_csv(output_csv, rows)
        print(f"scored_entities stt_model={model_id} path={score_path}", flush=True)

    write_entity_results_csv(output_csv, rows)
    print(f"wrote {output_csv}")
    return output_csv


def score_wer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 4: score WER directly from STT transcripts.")
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--transcripts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the parent of --transcripts-dir.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Defaults to results.csv in --output-dir.")
    parser.add_argument("--scores-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def score_wer_main(argv: list[str] | None = None) -> None:
    args = score_wer_parser().parse_args(argv)
    run_score_wer(args)


def run_score_wer(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    metadata_path = args.metadata or default_metadata_path(dataset_root)
    transcripts_dir = args.transcripts_dir.resolve()
    output_dir = (args.output_dir or default_run_output_dir(transcripts_dir, "transcripts")).resolve()
    output_csv = args.output_csv or output_dir / "results.csv"
    scores_dir = args.scores_dir
    datapoints = apply_limit(read_metadata(metadata_path), args.limit)
    transcript_files = row_data_files_by_model(transcripts_dir, "transcripts")
    if not transcript_files:
        raise RuntimeError(f"No transcript JSON files found in {transcripts_dir}.")
    sort_model_artifacts(transcript_files)

    all_scores = []
    model_ids = []
    rows = []
    for model_id, path in transcript_files:
        transcript_rows = read_complete_rows(path, datapoints, artifact_name="transcript")
        rows_by_audio_id = {row["audio_id"]: row for row in transcript_rows}
        scores = [score_wer(datapoint, model_id, rows_by_audio_id[datapoint["audio_id"]]) for datapoint in datapoints]
        score_path = scores_dir / f"{model_id}.json" if scores_dir else model_artifact_path(output_dir, model_id, "wer_scores")
        write_json_rows(score_path, scores, aggregate=aggregate_wer_score_rows(scores), run_metadata=read_json_run_metadata(path))
        all_scores.extend(scores)
        model_ids.append(model_id)
        rows = aggregate_wer_scores(all_scores, model_ids, len(datapoints))
        write_wer_results_csv(output_csv, rows)
        print(f"scored_wer stt_model={model_id} path={score_path}", flush=True)

    write_wer_results_csv(output_csv, rows)
    print(f"wrote {output_csv}")
    return output_csv
