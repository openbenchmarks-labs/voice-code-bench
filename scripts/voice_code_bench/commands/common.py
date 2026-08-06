from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from voice_code_bench.io import (
    parse_artifact_row,
    read_json_rows,
    read_metadata,
    read_metadata_rows,
    write_json_rows,
    write_metadata_jsonl,
)
from voice_code_bench.models import STT_MODEL_IDS

T = TypeVar("T")
R = TypeVar("R")


def apply_limit(datapoints: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return datapoints
    validate_positive(limit, "--limit")
    return datapoints[:limit]


def validate_positive(value: int, flag_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{flag_name} must be positive.")


def resolve_concurrency(value: int | None, default: int, flag_name: str) -> int:
    if value is not None:
        validate_positive(value, flag_name)
        return value
    validate_positive(default, flag_name)
    return default


def run_parallel_tasks(
    items: list[T],
    *,
    concurrency: int,
    task: Callable[[T], R],
    item_label: Callable[[T], str],
    finished_message: Callable[[str, R], str],
) -> list[R]:
    results: list[R] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(task, item): item_label(item) for item in items}
        for future in as_completed(futures):
            label = futures[future]
            result = future.result()
            results.append(result)
            print(finished_message(label, result), flush=True)
    return results


def metadata_order(datapoints: list[dict[str, Any]]) -> dict[str, int]:
    return {datapoint["audio_id"]: index for index, datapoint in enumerate(datapoints)}


def sort_model_artifacts(files: list[tuple[str, Path]]) -> None:
    known_order = {model_id: index for index, model_id in enumerate(STT_MODEL_IDS)}
    files.sort(key=lambda item: (known_order.get(item[0], 10_000), item[0]))


def read_complete_rows(path: Path, datapoints: list[dict[str, Any]], *, artifact_name: str) -> list[dict[str, Any]]:
    rows = [
        parse_artifact_row(path, row, index=index, artifact_name=artifact_name)
        for index, row in enumerate(read_json_rows(path), start=1)
    ]
    rows_by_audio_id = {row["audio_id"]: row for row in rows}
    missing = [datapoint["audio_id"] for datapoint in datapoints if datapoint["audio_id"] not in rows_by_audio_id]
    if missing:
        raise RuntimeError(f"{path} is missing {len(missing)} {artifact_name} row(s); first_missing={missing[0]}.")
    complete_rows = []
    for datapoint in datapoints:
        row = rows_by_audio_id[datapoint["audio_id"]]
        complete_rows.append(row)
    return complete_rows


def resume_rows_by_audio_id(path: Path, *, resume: bool, artifact_name: str | None = None) -> dict[str, dict[str, Any]]:
    if not resume or not path.exists():
        return {}
    rows = read_json_rows(path)
    if artifact_name is not None:
        rows = [
            parse_artifact_row(path, row, index=index, artifact_name=artifact_name)
            for index, row in enumerate(rows, start=1)
        ]
    return {row["audio_id"]: row for row in rows}


def ordered_artifact_rows(rows_by_audio_id: dict[str, dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    return sorted(rows_by_audio_id.values(), key=lambda row: order.get(row["audio_id"], 10**9))


def write_ordered_json_rows(
    path: Path,
    rows_by_audio_id: dict[str, dict[str, Any]],
    order: dict[str, int],
    *,
    aggregate: Any,
    run_metadata: dict[str, Any] | None,
) -> None:
    rows = ordered_artifact_rows(rows_by_audio_id, order)
    write_json_rows(path, rows, aggregate=aggregate(rows), run_metadata=run_metadata)


def metadata_path_for_available_audio(metadata_path: Path, dataset_root: Path, output_dir: Path) -> Path:
    datapoints = read_metadata(metadata_path)
    available_audio_ids = {
        datapoint["audio_id"] for datapoint in datapoints if (dataset_root / datapoint["audio_path"]).is_file()
    }
    missing_count = len(datapoints) - len(available_audio_ids)
    if missing_count == 0:
        print(f"metadata_audio_check available={len(datapoints)} missing=0", flush=True)
        return metadata_path
    if not available_audio_ids:
        raise RuntimeError(f"No metadata rows have local audio files under {dataset_root / 'data' / 'audio'}.")

    filtered_metadata = filter_metadata_items(metadata_path, available_audio_ids)
    filtered_path = output_dir / "metadata.available-audio.jsonl"
    write_metadata_jsonl(filtered_path, filtered_metadata)
    print(
        "metadata_audio_check "
        f"available={len(available_audio_ids)} missing={missing_count} "
        f"filtered_metadata={filtered_path}",
        flush=True,
    )
    return filtered_path


def filter_metadata_items(metadata_path: Path, audio_ids: set[str]) -> list[dict[str, Any]]:
    return [
        row
        for row in read_metadata_rows(metadata_path)
        if (row.get("audio_id") or row.get("id")) in audio_ids
    ]


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_evaluation_date(raw_value: str | None) -> str:
    if raw_value is None:
        return datetime.now(timezone.utc).date().isoformat()
    value = raw_value.strip()
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("--evaluation-date must use ISO format YYYY-MM-DD.") from exc
    return value
