from __future__ import annotations

import contextlib
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

from .commands.common import apply_limit, metadata_order, read_complete_rows
from .commands.extract_entities import write_entity_match_rows
from .http import HttpRequestError, get_bytes, get_json, post_json, post_multipart_json
from .io import (
    default_metadata_path,
    load_secret_file,
    model_artifact_path,
    read_json_run_metadata,
    read_metadata,
    require_secret,
    row_data_files_by_model,
)
from .verifier import (
    VERIFIER_CACHE_MODES,
    VerifierCache,
    VerifierConfig,
    build_cache_entry,
    parse_cached_entity_matches,
    parse_llm_entity_matches,
    stable_digest,
    verifier_cache_key,
    verifier_input,
)

BATCH_MANIFEST_SCHEMA_VERSION = 1
MAX_BATCH_ATTEMPTS = 3
OPENAI_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
OPENAI_IN_FLIGHT_STATUSES = {"validating", "in_progress", "finalizing"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n" for v in values), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must be an object.")
        rows.append(value)
    return rows


@contextlib.contextmanager
def manifest_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "manifest.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def batch_state_dir(run_dir: Path, verifier_id: str) -> Path:
    return run_dir / "verifier-batches" / verifier_id


def manifest_path(run_dir: Path, verifier_id: str) -> Path:
    return batch_state_dir(run_dir, verifier_id) / "manifest.json"


def load_manifest(run_dir: Path, verifier_id: str) -> dict[str, Any]:
    path = manifest_path(run_dir, verifier_id)
    if not path.is_file():
        raise RuntimeError(f"No batch plan at {path}; run --plan first.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != BATCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Invalid batch manifest at {path}.")
    return value


def save_manifest(run_dir: Path, verifier_id: str, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path(run_dir, verifier_id), manifest)


def batch_api_base(config: VerifierConfig) -> str:
    suffix = "/chat/completions"
    if not config.endpoint_or_api.endswith(suffix):
        raise ValueError("OpenAI Batch verifier endpoint must end with /chat/completions.")
    return config.endpoint_or_api[: -len(suffix)]


class OpenAIBatchClient:
    def __init__(self, api_key: str, config: VerifierConfig):
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.base = batch_api_base(config)
        self.timeout = config.timeout_seconds

    def upload(self, path: Path) -> dict[str, Any]:
        return post_multipart_json(
            f"{self.base}/files", {"purpose": "batch"}, "file", path,
            self.headers, file_content_type="application/jsonl", timeout=self.timeout,
        )

    def submit(self, input_file_id: str, completion_window: str, metadata: dict[str, str]) -> dict[str, Any]:
        return post_json(
            f"{self.base}/batches",
            {"input_file_id": input_file_id, "endpoint": "/v1/chat/completions", "completion_window": completion_window, "metadata": metadata},
            self.headers, timeout=self.timeout, max_attempts=1,
        )

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return get_json(f"{self.base}/batches/{batch_id}", self.headers, timeout=self.timeout)

    def find_by_submission_key(self, submission_key: str) -> dict[str, Any] | None:
        after: str | None = None
        for _ in range(20):
            query = {"limit": "100"}
            if after:
                query["after"] = after
            page = get_json(f"{self.base}/batches?{urlencode(query)}", self.headers, timeout=self.timeout)
            data = page.get("data", []) if isinstance(page, dict) else []
            for item in data:
                if isinstance(item, dict) and item.get("metadata", {}).get("vcb_submission_key") == submission_key:
                    return item
            if not page.get("has_more") or not data:
                return None
            after = data[-1].get("id")
        return None

    def download(self, file_id: str) -> bytes:
        return get_bytes(f"{self.base}/files/{file_id}/content", self.headers, timeout=self.timeout)


def validate_batch_config(config: VerifierConfig) -> None:
    if config.provider != "openai" or config.transport != "openai_batch":
        raise ValueError(f"Verifier {config.id} is not an OpenAI Batch verifier.")
    if config.datapoints_per_request != 10:
        raise ValueError(f"Verifier {config.id} must use the versioned group size of 10.")
    if config.request_options.get("reasoning_effort") != "medium":
        raise ValueError(f"Verifier {config.id} must use reasoning_effort=medium.")
    if config.completion_window != "24h":
        raise ValueError(f"Verifier {config.id} must use completion_window=24h.")


def grouped_user_input(datapoints: list[dict[str, Any]], transcript_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["audio_id"]: row for row in transcript_rows}
    return {"datapoints": [verifier_input(dp, by_id[dp["audio_id"]]["model_transcript"]) for dp in datapoints]}


def build_grouped_request(config: VerifierConfig, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": config.model,
        **config.request_options,
        "response_format": config.response_format,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": json.dumps({"datapoints": inputs}, ensure_ascii=False)},
        ],
    }


def custom_id(model_id: str, attempt: int, group_index: int, ids: list[str]) -> str:
    digest = stable_digest({"model": model_id, "attempt": attempt, "group": group_index, "ids": ids})[:16]
    return f"vcb-{attempt:02d}-{group_index:04d}-{digest}"


def make_attempt(
    state_dir: Path,
    model_id: str,
    attempt_number: int,
    unresolved_ids: list[str],
    inputs_by_id: dict[str, dict[str, Any]],
    config: VerifierConfig,
    *,
    write_request: bool = True,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    size = config.datapoints_per_request
    for group_index, start in enumerate(range(0, len(unresolved_ids), size)):
        ids = unresolved_ids[start : start + size]
        cid = custom_id(model_id, attempt_number, group_index, ids)
        groups.append({"custom_id": cid, "datapoint_ids": ids})
        lines.append({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": build_grouped_request(config, [inputs_by_id[value] for value in ids]),
        })
    relative = Path(model_id) / f"attempt-{attempt_number:02d}-requests.jsonl"
    if write_request:
        write_jsonl(state_dir / relative, lines)
    submission_key = stable_digest({"verifier": config.id, "model": model_id, "attempt": attempt_number, "groups": groups})[:32]
    return {
        "number": attempt_number,
        "state": "built",
        "request_file": str(relative),
        "request_count": len(lines),
        "datapoint_count": len(unresolved_ids),
        "groups": groups,
        "submission_key": submission_key,
        "input_file_id": None,
        "batch_id": None,
        "output_file_id": None,
        "error_file_id": None,
        "openai_status": None,
        "created_at": utc_now(),
        "errors": [],
    }


def parse_grouped_response(value: Any, expected_ids: list[str], datapoints_by_id: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict) or not isinstance(value.get("datapoints"), list):
        raise ValueError("Expected grouped verifier response with a datapoints array.")
    expected = set(expected_ids)
    found: dict[str, list[dict[str, Any]]] = {}
    for item in value["datapoints"]:
        if not isinstance(item, dict):
            raise ValueError("Grouped verifier datapoint must be an object.")
        datapoint_id = item.get("datapoint_id")
        if not isinstance(datapoint_id, str) or datapoint_id not in expected:
            raise ValueError(f"Unknown datapoint_id in grouped response: {datapoint_id!r}.")
        if datapoint_id in found:
            raise ValueError(f"Duplicate datapoint_id in grouped response: {datapoint_id}.")
        found[datapoint_id] = parse_llm_entity_matches({"entities": item.get("entities")}, datapoints_by_id[datapoint_id]["entities"])
    missing = [value for value in expected_ids if value not in found]
    if missing:
        raise ValueError(f"Grouped verifier response omitted datapoint_id values: {missing}.")
    return found


def plan_batches(
    run_dir: Path,
    dataset_root: Path,
    metadata_path: Path | None,
    config: VerifierConfig,
    *,
    model_ids: list[str] | None = None,
    limit: int | None = None,
    cache_path: Path | None = None,
    cache_mode: str = "live",
) -> dict[str, Any]:
    validate_batch_config(config)
    if cache_mode not in VERIFIER_CACHE_MODES:
        raise ValueError(f"Unsupported verifier cache mode: {cache_mode}")
    if cache_mode != "live" and cache_path is None:
        raise ValueError(f"--verifier-mode={cache_mode} requires --verifier-cache.")
    datapoints = apply_limit(read_metadata(metadata_path or default_metadata_path(dataset_root)), limit)
    datapoints_by_id = {dp["audio_id"]: dp for dp in datapoints}
    files = row_data_files_by_model(run_dir, "transcripts")
    if model_ids:
        wanted = set(model_ids)
        files = [item for item in files if item[0] in wanted]
        missing_models = wanted - {item[0] for item in files}
        if missing_models:
            raise RuntimeError(f"Missing completed transcripts for models: {sorted(missing_models)}")
    if not files:
        raise RuntimeError(f"No transcript JSON files found under {run_dir}.")
    cache = VerifierCache(cache_path) if cache_path else None
    state_dir = batch_state_dir(run_dir, config.id)
    path = manifest_path(run_dir, config.id)
    write_requests = not path.exists()
    models: dict[str, Any] = {}
    for model_id, transcript_path in files:
        transcript_rows = read_complete_rows(transcript_path, datapoints, artifact_name="transcript")
        inputs = grouped_user_input(datapoints, transcript_rows)["datapoints"]
        inputs_by_id = {item["datapoint_id"]: item for item in inputs}
        resolved: dict[str, Any] = {}
        pending: list[str] = []
        for dp in datapoints:
            request_input = inputs_by_id[dp["audio_id"]]
            key = verifier_cache_key(config, request_input)
            cached = cache.get(key) if cache is not None and cache_mode in {"replay", "live-fill"} else None
            if cached is not None:
                resolved[dp["audio_id"]] = {"entity_matches": parse_cached_entity_matches(cached, dp), "source": "cache"}
            elif cache_mode == "replay":
                raise RuntimeError(f"Verifier replay cache miss for audio_id={dp['audio_id']} stt_model={model_id} verifier_id={config.id}.")
            else:
                pending.append(dp["audio_id"])
        model: dict[str, Any] = {
            "model_id": model_id,
            "state": "planned",
            "transcript_path": str(transcript_path.resolve()),
            "transcript_digest": stable_digest(transcript_path.read_text(encoding="utf-8")),
            "source_run_metadata": read_json_run_metadata(transcript_path),
            "datapoint_ids": [dp["audio_id"] for dp in datapoints],
            "inputs": inputs_by_id,
            "resolved": resolved,
            "unresolved_ids": pending,
            "attempts": [],
            "usage": {},
            "errors": [],
        }
        if pending:
            model["attempts"].append(make_attempt(state_dir, model_id, 1, pending, inputs_by_id, config, write_request=write_requests))
            model["state"] = "built"
        else:
            model["state"] = "completed"
        models[model_id] = model
    manifest = {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "run_dir": str(run_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "metadata_path": str((metadata_path or default_metadata_path(dataset_root)).resolve()),
        "metadata_digest": stable_digest([{"audio_id": dp["audio_id"], "entities": dp["entities"]} for dp in datapoints]),
        "verifier_id": config.id,
        "verifier_config_digest": config.digest,
        "transport": config.transport,
        "datapoints_per_request": config.datapoints_per_request,
        "reasoning_effort": config.request_options["reasoning_effort"],
        "completion_window": config.completion_window,
        "cache_mode": cache_mode,
        "cache_path": str(cache_path.resolve()) if cache_path else None,
        "models": models,
    }
    if path.exists():
        existing = load_manifest(run_dir, config.id)
        immutable = ("metadata_digest", "verifier_config_digest", "cache_mode", "cache_path")
        same = all(existing.get(key) == manifest.get(key) for key in immutable) and {
            key: value.get("transcript_digest") for key, value in existing.get("models", {}).items()
        } == {key: value["transcript_digest"] for key, value in models.items()}
        if not same:
            raise RuntimeError(f"Existing plan at {path} has different inputs or configuration; use a new --run-dir.")
        return existing
    save_manifest(run_dir, config.id, manifest)
    return manifest


def client_for(secrets: dict[str, str], config: VerifierConfig) -> OpenAIBatchClient:
    return OpenAIBatchClient(require_secret(secrets, "OPENAI_API_KEY"), config)


def current_attempt(model: dict[str, Any]) -> dict[str, Any]:
    if not model["attempts"]:
        raise RuntimeError(f"Model {model['model_id']} has no batch attempt.")
    return model["attempts"][-1]


def in_flight_count(manifest: dict[str, Any]) -> int:
    count = 0
    for model in manifest["models"].values():
        if model["attempts"] and current_attempt(model).get("state") in {"submitting", "submitted"}:
            count += 1
    return count


def reconcile_batches(
    run_dir: Path,
    config: VerifierConfig,
    secrets: dict[str, str],
    *,
    max_in_flight: int = 4,
    client: OpenAIBatchClient | None = None,
) -> tuple[dict[str, Any], bool]:
    validate_batch_config(config)
    state_dir = batch_state_dir(run_dir, config.id)
    with manifest_lock(state_dir):
        manifest = load_manifest(run_dir, config.id)
        if manifest["verifier_config_digest"] != config.digest:
            raise RuntimeError("Batch plan verifier configuration no longer matches; use a new run directory.")
        api = client

        def require_api() -> OpenAIBatchClient:
            nonlocal api
            if api is None:
                api = client_for(secrets, config)
            return api
        progressed = False
        for model in manifest["models"].values():
            if model["state"] in {"collected", "failed"}:
                continue
            if not model["unresolved_ids"]:
                collect_model(run_dir, manifest, model, config)
                progressed = True
                continue
            attempt = current_attempt(model)
            if attempt["state"] == "built":
                upload = require_api().upload(state_dir / attempt["request_file"])
                attempt["input_file_id"] = upload["id"]
                attempt["state"] = "uploaded"
                model["state"] = "uploaded"
                save_manifest(run_dir, config.id, manifest)
                progressed = True
            if attempt["state"] in {"uploaded", "submitting"}:
                recovered = require_api().find_by_submission_key(attempt["submission_key"])
                if recovered is not None:
                    record_submitted(attempt, model, recovered)
                    save_manifest(run_dir, config.id, manifest)
                    progressed = True
                elif in_flight_count(manifest) >= max_in_flight:
                    continue
                else:
                    attempt["state"] = "submitting"
                    model["state"] = "submitted"
                    save_manifest(run_dir, config.id, manifest)
                    try:
                        submitted = require_api().submit(
                            attempt["input_file_id"], config.completion_window or "24h",
                            {"vcb_submission_key": attempt["submission_key"], "vcb_model_id": model["model_id"], "vcb_verifier_id": config.id},
                        )
                    except HttpRequestError as exc:
                        attempt["state"] = "uploaded"
                        model["state"] = "uploaded"
                        if exc.status_code == 429:
                            attempt["deferred_reason"] = str(exc)
                            save_manifest(run_dir, config.id, manifest)
                            continue
                        save_manifest(run_dir, config.id, manifest)
                        raise
                    record_submitted(attempt, model, submitted)
                    save_manifest(run_dir, config.id, manifest)
                    progressed = True
            if attempt["state"] == "submitted":
                remote = require_api().retrieve(attempt["batch_id"])
                status = remote.get("status")
                attempt["openai_status"] = status
                attempt["output_file_id"] = remote.get("output_file_id")
                attempt["error_file_id"] = remote.get("error_file_id")
                if isinstance(remote.get("usage"), dict):
                    attempt["usage"] = remote["usage"]
                    merge_usage(model["usage"], remote["usage"])
                if status not in OPENAI_TERMINAL_STATUSES:
                    save_manifest(run_dir, config.id, manifest)
                    continue
                attempt["state"] = "completed" if status == "completed" else "failed"
                model["state"] = attempt["state"]
                save_manifest(run_dir, config.id, manifest)
                progressed = True
            if attempt["state"] == "completed":
                collect_attempt(state_dir, model, attempt, require_api(), manifest, config)
                save_manifest(run_dir, config.id, manifest)
                progressed = True
            if attempt["state"] == "failed":
                retry_or_fail(state_dir, model, config, f"OpenAI Batch ended with status {attempt.get('openai_status')}")
                save_manifest(run_dir, config.id, manifest)
                progressed = True
            if not model["unresolved_ids"] and model["state"] != "collected":
                collect_model(run_dir, manifest, model, config)
                save_manifest(run_dir, config.id, manifest)
                progressed = True
        return manifest, progressed


def record_submitted(attempt: dict[str, Any], model: dict[str, Any], remote: dict[str, Any]) -> None:
    attempt["batch_id"] = remote["id"]
    attempt["openai_status"] = remote.get("status")
    attempt["state"] = "submitted"
    attempt.pop("deferred_reason", None)
    model["state"] = "submitted"


def merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def collect_attempt(
    state_dir: Path,
    model: dict[str, Any],
    attempt: dict[str, Any],
    api: OpenAIBatchClient,
    manifest: dict[str, Any],
    config: VerifierConfig,
) -> None:
    output_rows: list[dict[str, Any]] = []
    for kind in ("output", "error"):
        file_id = attempt.get(f"{kind}_file_id")
        if not file_id:
            continue
        relative = Path(model["model_id"]) / f"attempt-{attempt['number']:02d}-{kind}.jsonl"
        path = state_dir / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(api.download(file_id))
            tmp.replace(path)
        attempt[f"{kind}_file"] = str(relative)
        if kind == "output":
            output_rows = read_jsonl(path)
    by_custom: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    duplicate_custom_ids: set[str] = set()
    expected_custom = {group["custom_id"] for group in attempt["groups"]}
    for row in output_rows:
        cid = row.get("custom_id")
        if cid not in expected_custom:
            unknown.append(str(cid))
        elif cid in by_custom or cid in duplicate_custom_ids:
            duplicate_custom_ids.add(str(cid))
            by_custom.pop(str(cid), None)
        elif isinstance(cid, str):
            by_custom[cid] = row
    datapoints_by_id = read_plan_datapoints(manifest)
    cache = VerifierCache(Path(manifest["cache_path"])) if manifest.get("cache_path") else None
    for group in attempt["groups"]:
        row = by_custom.get(group["custom_id"])
        if row is None:
            continue
        try:
            response = row["response"]
            if response.get("status_code") != 200:
                raise ValueError(f"HTTP {response.get('status_code')}")
            body = response["body"]
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            matches = parse_grouped_response(parsed, group["datapoint_ids"], datapoints_by_id)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
            attempt["errors"].append(f"custom_id={group['custom_id']}: {exc}")
            continue
        usage = body.get("usage")
        if "usage" not in attempt and isinstance(usage, dict):
            merge_usage(model["usage"], usage)
        for datapoint_id, entity_matches in matches.items():
            model["resolved"][datapoint_id] = {"entity_matches": entity_matches, "source": "batch", "batch_id": attempt["batch_id"]}
            if cache is not None:
                request_input = model["inputs"][datapoint_id]
                response_value = {"entities": entity_matches}
                cache.put(build_cache_entry(config, request_input, verifier_cache_key(config, request_input), response_value))
    if unknown:
        attempt["errors"].append(f"Unknown or duplicate custom_id values: {unknown}")
    if duplicate_custom_ids:
        attempt["errors"].append(f"Duplicate custom_id values: {sorted(duplicate_custom_ids)}")
    model["unresolved_ids"] = [value for value in model["datapoint_ids"] if value not in model["resolved"]]
    attempt["state"] = "collected"
    if model["unresolved_ids"]:
        retry_or_fail(state_dir, model, config, f"{len(model['unresolved_ids'])} datapoint(s) unresolved after collection")
    else:
        model["state"] = "completed"


def retry_or_fail(state_dir: Path, model: dict[str, Any], config: VerifierConfig, reason: str) -> None:
    attempt = current_attempt(model)
    attempt["errors"].append(reason)
    if attempt["number"] >= MAX_BATCH_ATTEMPTS:
        model["state"] = "failed"
        model["errors"].append(reason)
        return
    next_number = attempt["number"] + 1
    model["attempts"].append(make_attempt(state_dir, model["model_id"], next_number, model["unresolved_ids"], model["inputs"], config))
    model["state"] = "built"


def read_plan_datapoints(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datapoints = read_metadata(Path(manifest["metadata_path"]))
    return {dp["audio_id"]: dp for dp in datapoints}


def collect_model(run_dir: Path, manifest: dict[str, Any], model: dict[str, Any], config: VerifierConfig) -> None:
    datapoints_by_id = read_plan_datapoints(manifest)
    order = metadata_order([datapoints_by_id[value] for value in model["datapoint_ids"]])
    transcript_rows = read_complete_rows(Path(model["transcript_path"]), [datapoints_by_id[value] for value in model["datapoint_ids"]], artifact_name="transcript")
    transcript_by_id = {row["audio_id"]: row for row in transcript_rows}
    rows: dict[str, dict[str, Any]] = {}
    for datapoint_id in model["datapoint_ids"]:
        transcript_row = transcript_by_id[datapoint_id]
        row = {
            "audio_id": datapoint_id,
            "model_transcript": transcript_row["model_transcript"],
            "entity_matches": model["resolved"][datapoint_id]["entity_matches"],
        }
        if "metadata" in transcript_row:
            row["metadata"] = transcript_row["metadata"]
        rows[datapoint_id] = row
    source_metadata = model.get("source_run_metadata")
    run_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    run_metadata["entity_verifier"] = {
        "id": config.id, "provider": config.provider, "model": config.model,
        "endpoint_or_api": config.endpoint_or_api, "config_digest": config.digest,
        "cache_mode": manifest["cache_mode"], "transport": config.transport,
        "datapoints_per_request": config.datapoints_per_request,
        "reasoning_effort": config.request_options["reasoning_effort"],
        "batch_ids": [attempt.get("batch_id") for attempt in model["attempts"] if attempt.get("batch_id")],
        "request_count": sum(attempt["request_count"] for attempt in model["attempts"]),
        "usage": model["usage"],
    }
    write_entity_match_rows(model_artifact_path(run_dir, model["model_id"], "entity_matches"), rows, order, run_metadata)
    model["state"] = "collected"


def all_terminal(manifest: dict[str, Any]) -> bool:
    return all(model["state"] in {"collected", "failed"} for model in manifest["models"].values())


def validate_manifest(run_dir: Path, config: VerifierConfig) -> None:
    manifest = load_manifest(run_dir, config.id)
    if manifest["verifier_config_digest"] != config.digest:
        raise RuntimeError("Verifier config digest does not match the plan.")
    datapoints = read_plan_datapoints(manifest)
    metadata_digest = stable_digest([{"audio_id": dp["audio_id"], "entities": dp["entities"]} for dp in datapoints.values() if dp["audio_id"] in next(iter(manifest["models"].values()))["datapoint_ids"]])
    if metadata_digest != manifest["metadata_digest"]:
        raise RuntimeError("Metadata no longer matches the batch plan.")
    for model_id, model in manifest["models"].items():
        if model["state"] != "collected":
            raise RuntimeError(f"stt_model={model_id} is {model['state']}, not collected.")
        transcript_path = Path(model["transcript_path"])
        if stable_digest(transcript_path.read_text(encoding="utf-8")) != model["transcript_digest"]:
            raise RuntimeError(f"stt_model={model_id} transcripts changed after planning.")
        output = model_artifact_path(run_dir, model_id, "entity_matches")
        rows = read_complete_rows(output, [datapoints[value] for value in model["datapoint_ids"]], artifact_name="entity-match")
        if len(rows) != len(model["datapoint_ids"]):
            raise RuntimeError(f"stt_model={model_id} output row count mismatch.")


def status_lines(manifest: dict[str, Any]) -> list[str]:
    lines = []
    total_usage: dict[str, Any] = {}
    for model_id, model in manifest["models"].items():
        attempt = model["attempts"][-1] if model["attempts"] else {}
        usage = model.get("usage", {})
        merge_usage(total_usage, usage)
        request_count = sum(item.get("request_count", 0) for item in model["attempts"])
        lines.append(
            f"{model_id}: state={model['state']} resolved={len(model['resolved'])}/{len(model['datapoint_ids'])} "
            f"unresolved={len(model['unresolved_ids'])} attempts={len(model['attempts'])} requests={request_count} "
            f"batch_id={attempt.get('batch_id') or '-'} openai_status={attempt.get('openai_status') or '-'} "
            f"usage={json.dumps(usage, sort_keys=True)}"
        )
    lines.append(f"total_usage={json.dumps(total_usage, sort_keys=True)}")
    return lines
