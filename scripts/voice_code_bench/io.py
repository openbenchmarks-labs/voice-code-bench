from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import ENTITY_TYPES

ROW_JSON_SCHEMA_VERSION = 1


def default_secret_path(dataset_root: Path) -> Path | None:
    from_env = os.environ.get("VOICE_CODE_BENCH_SECRET", "").strip()
    if from_env:
        return Path(from_env)
    for candidate in [dataset_root / "scripts" / ".secret", Path.cwd() / "scripts" / ".secret"]:
        if candidate.exists():
            return candidate
    return None


def load_secret_file(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("\"'")
    return out


def secret_value(secrets: dict[str, str], key: str) -> str | None:
    value = os.environ.get(key, "").strip() or secrets.get(key, "").strip()
    return value or None


def require_secret(secrets: dict[str, str], key: str) -> str:
    value = secret_value(secrets, key)
    if not value:
        raise RuntimeError(f"Missing {key}. Set it in the environment or pass --secret-path.")
    return value


def read_metadata(path: Path) -> list[dict[str, Any]]:
    raw_rows = read_metadata_rows(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Metadata row {index} must be an object.")
        audio_id = row.get("audio_id") or row.get("id")
        if not isinstance(audio_id, str) or not audio_id:
            raise ValueError(f"Metadata row {index} is missing audio_id/id.")
        if audio_id in seen:
            raise ValueError(f"Duplicate audio_id in metadata row {index}: {audio_id}")
        if not isinstance(row.get("file_name"), str) or not row["file_name"]:
            raise ValueError(f"Metadata row {index} is missing file_name.")
        duration = row.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"Metadata row {index} has invalid duration: {duration!r}.")
        validate_transcripts(row.get("transcripts"), audio_id=audio_id, line_number=index)
        if not isinstance(row.get("entities"), list):
            raise ValueError(f"Metadata row {index} is missing entities.")
        if not isinstance(row.get("domain"), str) or not row["domain"]:
            raise ValueError(f"Metadata row {index} is missing domain.")
        if not isinstance(row.get("scenario"), str) or not row["scenario"]:
            raise ValueError(f"Metadata row {index} is missing scenario.")
        if row.get("difficulty") not in {"light", "standard", "dense", "stress"}:
            raise ValueError(f"Metadata row {index} has invalid difficulty: {row.get('difficulty')!r}.")
        seen.add(audio_id)
        rows.append(metadata_row_to_datapoint(row, line_number=index))
    return rows


def read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object.")
            rows.append(row)
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError(
            f"{path} must be metadata JSONL or a legacy metadata JSON object with an items array."
        )
    return data["items"]


def write_metadata_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def default_metadata_path(dataset_root: Path) -> Path:
    return dataset_root / "data" / "metadata.jsonl"


def metadata_row_to_datapoint(row: dict[str, Any], *, line_number: int | None = None) -> dict[str, Any]:
    out = dict(row)
    audio_id = out.get("audio_id") or out.get("id")
    out["audio_id"] = audio_id
    validate_transcripts(out.get("transcripts"), audio_id=str(audio_id), line_number=line_number)
    out["transcript"] = out["transcripts"]["canonical"]
    out["audio_path"] = str(Path("data") / out["file_name"])
    out["entities"] = normalize_entities(out["entities"], audio_id=str(audio_id), line_number=line_number)
    entity_types = [entity["type"] for entity in out["entities"]]
    unique_entity_types = list(dict.fromkeys(entity_types))
    if out.get("entity_count") != len(out["entities"]):
        location = metadata_location(audio_id=str(audio_id), line_number=line_number)
        raise ValueError(f"{location} has entity_count={out.get('entity_count')!r}, expected {len(out['entities'])}.")
    if out.get("entity_types") != unique_entity_types:
        location = metadata_location(audio_id=str(audio_id), line_number=line_number)
        raise ValueError(f"{location} has entity_types={out.get('entity_types')!r}, expected {unique_entity_types!r}.")
    return out


def validate_transcripts(transcripts: Any, *, audio_id: str, line_number: int | None) -> None:
    location = metadata_location(audio_id=audio_id, line_number=line_number)
    if not isinstance(transcripts, dict):
        raise ValueError(f"{location} is missing transcripts.")
    for layer in ("template", "acoustic", "canonical"):
        if not isinstance(transcripts.get(layer), str) or not transcripts[layer]:
            raise ValueError(f"{location} is missing transcripts.{layer}.")


def normalize_entities(entities: list[Any], *, audio_id: str, line_number: int | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw_entity in enumerate(entities):
        location = f"{metadata_location(audio_id=audio_id, line_number=line_number)} entity #{index}"
        if not isinstance(raw_entity, dict):
            raise ValueError(f"{location} must be an object.")
        entity_id = raw_entity.get("id")
        entity_type = raw_entity.get("type")
        role = raw_entity.get("role")
        acoustic = raw_entity.get("acoustic")
        canonical = raw_entity.get("canonical")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"{location} is missing id.")
        if entity_id in seen_ids:
            raise ValueError(f"{location} duplicates entity id {entity_id!r}.")
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"{location} has invalid type {entity_type!r}.")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{location} is missing role.")
        if not isinstance(acoustic, str) or not acoustic:
            raise ValueError(f"{location} is missing acoustic.")
        if not isinstance(canonical, str) or not canonical:
            raise ValueError(f"{location} is missing canonical.")
        seen_ids.add(entity_id)
        out.append(
            {
                "id": entity_id,
                "type": entity_type,
                "role": role,
                "acoustic": acoustic,
                "canonical": canonical,
            }
        )
    return out


def metadata_location(*, audio_id: str, line_number: int | None) -> str:
    if line_number is None:
        return f"metadata row audio_id={audio_id}"
    return f"metadata row {line_number} audio_id={audio_id}"


def read_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": []}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError(f"{path} must be a JSON object with an items array.")
    rows = data["items"]
    validate_unique_row_ids(path, rows)
    return data


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    return read_json_artifact(path)["items"]


def parse_artifact_row(path: Path, row: Any, *, index: int, artifact_name: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{path} {artifact_name} item {index} must be an object.")
    audio_id = row.get("audio_id")
    if not isinstance(audio_id, str) or not audio_id:
        raise ValueError(f"{path} {artifact_name} item {index} is missing audio_id.")
    if artifact_name in {"transcript", "entity-match"} and not isinstance(row.get("model_transcript"), str):
        raise ValueError(f"{path} {artifact_name} row audio_id={audio_id} is missing model_transcript.")
    if "metadata" in row and not isinstance(row["metadata"], dict):
        raise ValueError(f"{path} {artifact_name} row audio_id={audio_id} metadata must be an object.")
    if artifact_name == "entity-match" and not isinstance(row.get("entity_matches"), list):
        raise ValueError(f"{path} entity-match row audio_id={audio_id} is missing entity_matches.")
    return row


def read_json_run_metadata(path: Path) -> dict[str, Any] | None:
    metadata = read_json_artifact(path).get("run_metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{path} run_metadata must be an object.")
    return metadata


def write_json_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    aggregate: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"schema_version": ROW_JSON_SCHEMA_VERSION}
    if run_metadata is not None:
        data["run_metadata"] = run_metadata
    data["aggregate"] = aggregate or {"row_count": len(rows)}
    data["items"] = rows
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def row_data_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def model_artifact_path(run_dir: Path, model_id: str, artifact_name: str) -> Path:
    return run_dir / model_id / f"{artifact_name}.json"


def default_run_output_dir(input_dir: Path, artifact_name: str) -> Path:
    if any(input_dir.glob(f"*/{artifact_name}.json")):
        return input_dir
    if (input_dir / f"{artifact_name}.json").exists():
        return input_dir.parent
    return input_dir.parent


def row_data_files_by_model(directory: Path, artifact_name: str) -> list[tuple[str, Path]]:
    direct_json_path = directory / f"{artifact_name}.json"
    if direct_json_path.is_file():
        return [(directory.name, direct_json_path)]

    nested_json_paths = sorted(path for path in directory.glob(f"*/{artifact_name}.json") if path.is_file())
    if nested_json_paths:
        return [(path.parent.name, path) for path in nested_json_paths]

    return [(path.stem, path) for path in row_data_files(directory)]


def validate_unique_row_ids(path: Path, rows: list[Any]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} item {index} must be an object.")
        row_id = row.get("audio_id") or row.get("datapoint_id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{path} item {index} is missing audio_id/datapoint_id.")
        if row_id in seen:
            raise ValueError(f"Duplicate row id in {path} item {index}: {row_id}")
        seen.add(row_id)
