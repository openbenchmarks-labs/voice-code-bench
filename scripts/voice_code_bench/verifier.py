from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .http import HttpRequestError, post_json
from .io import require_secret
from .models import DEFAULT_VERIFIER_ID

VERIFIER_CACHE_MODES = ("live", "replay", "live-fill")
GROUPED_SYNC_MAX_ATTEMPTS = 3
VERIFIER_CONFIG_DIR = Path(__file__).with_name("verifiers")
VERIFIER_CACHE_SCHEMA_VERSION = 1

_CACHE_LOCKS: dict[Path, threading.Lock] = {}
_CACHE_LOCKS_LOCK = threading.Lock()


@dataclass(frozen=True)
class VerifierConfig:
    id: str
    provider: str
    model: str
    endpoint_or_api: str
    timeout_seconds: float
    request_options: dict[str, Any]
    system_prompt: str
    response_format: dict[str, Any]
    transport: str = "sync"
    datapoints_per_request: int = 1
    completion_window: str | None = None

    @property
    def digest(self) -> str:
        payload = {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "endpoint_or_api": self.endpoint_or_api,
            "timeout_seconds": self.timeout_seconds,
            "request_options": self.request_options,
            "system_prompt": self.system_prompt,
            "response_format": self.response_format,
            "transport": self.transport,
            "datapoints_per_request": self.datapoints_per_request,
            "completion_window": self.completion_window,
        }
        return stable_digest(payload)


class VerifierCache:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self._lock = cache_file_lock(self.path)
        self._entries_by_key = self._read_entries()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            with cache_file_process_lock(self.path):
                self._entries_by_key = self._read_entries()
                entry = self._entries_by_key.get(cache_key)
                return dict(entry) if entry else None

    def put(self, entry: dict[str, Any]) -> None:
        cache_key = entry.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("Verifier cache entry is missing cache_key.")
        with self._lock:
            with cache_file_process_lock(self.path):
                self._entries_by_key = self._read_entries()
                self._entries_by_key[cache_key] = dict(entry)
                self._write_entries()

    def _read_entries(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != VERIFIER_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{self.path} must be a verifier cache object with schema_version={VERIFIER_CACHE_SCHEMA_VERSION}.")
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"{self.path} verifier cache is missing an entries array.")
        out: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"{self.path} verifier cache entry {index} must be an object.")
            cache_key = entry.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                raise ValueError(f"{self.path} verifier cache entry {index} is missing cache_key.")
            out[cache_key] = entry
        return out

    def _write_entries(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = [self._entries_by_key[key] for key in sorted(self._entries_by_key)]
        data = {"schema_version": VERIFIER_CACHE_SCHEMA_VERSION, "entries": entries}
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)


def cache_file_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _CACHE_LOCKS_LOCK:
        lock = _CACHE_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[resolved] = lock
        return lock


@contextlib.contextmanager
def cache_file_process_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_verifier_config(verifier_id: str = DEFAULT_VERIFIER_ID) -> VerifierConfig:
    value = verifier_id.strip()
    if not value:
        raise ValueError("Verifier id must be non-empty.")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"Invalid verifier id: {verifier_id!r}")
    path = VERIFIER_CONFIG_DIR / f"{value}.json"
    if not path.is_file():
        raise ValueError(f"Unknown verifier id {value!r}; expected config at {path}.")
    return parse_verifier_config(json.loads(path.read_text(encoding="utf-8")), path=path)


def parse_verifier_config(data: Any, *, path: Path | None = None) -> VerifierConfig:
    location = str(path) if path is not None else "verifier config"
    if not isinstance(data, dict):
        raise ValueError(f"{location} must be a JSON object.")
    if data.get("schema_version") != 1:
        raise ValueError(f"{location} must use schema_version=1.")
    verifier_id = required_string(data, "id", location)
    provider = required_string(data, "provider", location)
    model = required_string(data, "model", location)
    endpoint_or_api = required_string(data, "endpoint_or_api", location)
    timeout_seconds = data.get("timeout_seconds")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError(f"{location} must define a positive timeout_seconds.")
    request_options = data.get("request_options", {})
    if not isinstance(request_options, dict):
        raise ValueError(f"{location} request_options must be an object.")
    system_prompt = data.get("system_prompt")
    if isinstance(system_prompt, list) and all(isinstance(line, str) for line in system_prompt):
        prompt = "\n".join(system_prompt)
    elif isinstance(system_prompt, str):
        prompt = system_prompt
    else:
        raise ValueError(f"{location} system_prompt must be a string or array of strings.")
    if not prompt.strip():
        raise ValueError(f"{location} system_prompt must be non-empty.")
    response_format = data.get("response_format")
    if not isinstance(response_format, dict):
        raise ValueError(f"{location} response_format must be an object.")
    transport = data.get("transport", "sync")
    if transport not in {"sync", "openai_batch"}:
        raise ValueError(f"{location} has unsupported transport {transport!r}.")
    datapoints_per_request = data.get("datapoints_per_request", 1)
    if isinstance(datapoints_per_request, bool) or not isinstance(datapoints_per_request, int) or datapoints_per_request <= 0:
        raise ValueError(f"{location} datapoints_per_request must be a positive integer.")
    completion_window = data.get("completion_window")
    if completion_window is not None and (not isinstance(completion_window, str) or not completion_window.strip()):
        raise ValueError(f"{location} completion_window must be a non-empty string.")
    return VerifierConfig(
        id=verifier_id,
        provider=provider,
        model=model,
        endpoint_or_api=endpoint_or_api,
        timeout_seconds=float(timeout_seconds),
        request_options=dict(request_options),
        system_prompt=prompt,
        response_format=response_format,
        transport=transport,
        datapoints_per_request=datapoints_per_request,
        completion_window=completion_window,
    )


def required_string(data: dict[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must define non-empty {key}.")
    return value


def verifier_config_for_args(verifier_id: str) -> VerifierConfig:
    return load_verifier_config(verifier_id)


def target_entity_payload(datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "target_index": index,
            "id": entity["id"],
            "type": entity["type"],
            "role": entity["role"],
            "acoustic": entity["acoustic"],
            "canonical": entity["canonical"],
        }
        for index, entity in enumerate(datapoint["entities"])
    ]


def verifier_input(datapoint: dict[str, Any], transcript: str) -> dict[str, Any]:
    return {
        "datapoint_id": datapoint["audio_id"],
        "target_entities": target_entity_payload(datapoint),
        "transcript": transcript,
    }


def verifier_cache_key(config: VerifierConfig, request_input: dict[str, Any]) -> str:
    return stable_digest(
        {
            "verifier_id": config.id,
            "config_digest": config.digest,
            "request_input": request_input,
        }
    )


def verify_entity_matches(
    datapoint: dict[str, Any],
    transcript: str,
    config: VerifierConfig,
    secrets: dict[str, str],
    *,
    cache: VerifierCache | None = None,
    cache_mode: str = "live",
) -> list[dict[str, Any]]:
    if cache_mode not in VERIFIER_CACHE_MODES:
        raise ValueError(f"Unsupported verifier cache mode: {cache_mode}")
    if cache_mode != "live" and cache is None:
        raise ValueError(f"--verifier-mode={cache_mode} requires --verifier-cache.")

    request_input = verifier_input(datapoint, transcript)
    cache_key = verifier_cache_key(config, request_input)
    if cache is not None and cache_mode in {"replay", "live-fill"}:
        cached = cache.get(cache_key)
        if cached is not None:
            return parse_cached_entity_matches(cached, datapoint)
    if cache_mode == "replay":
        raise RuntimeError(f"Verifier replay cache miss for audio_id={datapoint['audio_id']} verifier_id={config.id}.")

    response = call_verifier_provider(config, datapoint, transcript, secrets)
    if cache is not None:
        cache.put(build_cache_entry(config, request_input, cache_key, response))
    return parse_response_entity_matches(response, datapoint)


def parse_cached_entity_matches(entry: dict[str, Any], datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    response = entry.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Verifier cache entry for audio_id={datapoint['audio_id']} is missing response object.")
    return parse_response_entity_matches(response, datapoint)


def parse_response_entity_matches(response: dict[str, Any], datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return parse_llm_entity_matches(response, datapoint["entities"])
    except ValueError as exc:
        raise ValueError(f"Invalid verifier response for audio_id={datapoint['audio_id']}: {exc}") from exc


def build_cache_entry(
    config: VerifierConfig,
    request_input: dict[str, Any],
    cache_key: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "verifier_id": config.id,
        "provider": config.provider,
        "model": config.model,
        "config_digest": config.digest,
        "datapoint_id": request_input["datapoint_id"],
        "transcript_sha256": stable_digest(request_input["transcript"]),
        "request_input": request_input,
        "response": response,
    }


def call_verifier_provider(
    config: VerifierConfig,
    datapoint: dict[str, Any],
    transcript: str,
    secrets: dict[str, str],
) -> dict[str, Any]:
    if config.provider == "openai":
        return call_openai_verifier(config, datapoint, transcript, secrets)
    raise ValueError(f"Unsupported verifier provider: {config.provider}")


def call_openai_verifier(
    config: VerifierConfig,
    datapoint: dict[str, Any],
    transcript: str,
    secrets: dict[str, str],
) -> dict[str, Any]:
    request = build_openai_verify_request(config, datapoint, transcript)
    response = post_json(
        config.endpoint_or_api,
        request,
        {"Authorization": f"Bearer {require_secret(secrets, 'OPENAI_API_KEY')}"},
        timeout=config.timeout_seconds,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"OpenAI verifier response did not include message content for audio_id={datapoint['audio_id']}.") from exc
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Verifier returned invalid JSON for audio_id={datapoint['audio_id']}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Verifier returned non-object JSON for audio_id={datapoint['audio_id']}.")
    return parsed


def build_openai_verify_request(config: VerifierConfig, datapoint: dict[str, Any], transcript: str) -> dict[str, Any]:
    request = {
        "model": config.model,
        **config.request_options,
        "response_format": config.response_format,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": json.dumps(verifier_input(datapoint, transcript), ensure_ascii=False)},
        ],
    }
    if "temperature" not in request:
        temperature = verifier_temperature(config.model)
        if temperature is not None:
            request["temperature"] = temperature
    return request


def build_grouped_verify_request(config: VerifierConfig, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": config.model,
        **config.request_options,
        "response_format": config.response_format,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": json.dumps({"datapoints": inputs}, ensure_ascii=False)},
        ],
    }


def verify_grouped_entity_matches(
    datapoints: list[dict[str, Any]],
    transcript_rows: list[dict[str, Any]],
    config: VerifierConfig,
    secrets: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    if config.provider != "openai" or config.transport != "sync":
        raise ValueError(f"Verifier {config.id} is not a synchronous OpenAI verifier.")
    expected_ids = [datapoint["audio_id"] for datapoint in datapoints]
    if len(expected_ids) > config.datapoints_per_request:
        raise ValueError(
            f"Verifier {config.id} accepts at most {config.datapoints_per_request} datapoints per request."
        )
    transcript_by_id = {row["audio_id"]: row for row in transcript_rows}
    inputs = [
        verifier_input(datapoint, transcript_by_id[datapoint["audio_id"]]["model_transcript"])
        for datapoint in datapoints
    ]
    last_error: Exception | None = None
    for _attempt in range(1, GROUPED_SYNC_MAX_ATTEMPTS + 1):
        try:
            response = call_grouped_openai_verifier(config, inputs, secrets)
            return parse_grouped_response(
                response,
                expected_ids,
                {datapoint["audio_id"]: datapoint for datapoint in datapoints},
            )
        except (ValueError, HttpRequestError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def call_grouped_openai_verifier(
    config: VerifierConfig,
    inputs: list[dict[str, Any]],
    secrets: dict[str, str],
) -> dict[str, Any]:
    response = post_json(
        config.endpoint_or_api,
        build_grouped_verify_request(config, inputs),
        {"Authorization": f"Bearer {require_secret(secrets, 'OPENAI_API_KEY')}"},
        timeout=config.timeout_seconds,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI grouped verifier response did not include message content.") from exc
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI grouped verifier returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI grouped verifier returned non-object JSON.")
    return parsed


def parse_grouped_response(
    value: Any,
    expected_ids: list[str],
    datapoints_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
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
        found[datapoint_id] = parse_llm_entity_matches(
            {"entities": item.get("entities")},
            datapoints_by_id[datapoint_id]["entities"],
        )
    missing = [datapoint_id for datapoint_id in expected_ids if datapoint_id not in found]
    if missing:
        raise ValueError(f"Grouped verifier response omitted datapoint_id values: {missing}.")
    return found


def verifier_temperature(model: str) -> int | None:
    if model.strip().lower().startswith("gpt-5"):
        return None
    return 0


def parse_llm_entity_matches(
    value: Any,
    target_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("entities"), list):
        raise ValueError("Expected verifier response with an entities array.")

    targets = []
    for index, entity in enumerate(target_entities):
        targets.append(
            {
                "target_index": index,
                "type": entity["type"],
                "canonical": entity["canonical"],
            }
        )
    by_index = {target["target_index"]: target for target in targets}
    out_by_index: dict[int, dict[str, Any]] = {}
    for item in value["entities"]:
        if not isinstance(item, dict):
            raise ValueError("Expected verifier entity item to be an object.")
        target_index = item.get("target_index")
        entity_type = item.get("type")
        canonical = item.get("canonical")
        present = item.get("present")
        evidence = item.get("evidence")
        reason = item.get("reason")
        if not isinstance(target_index, int) or target_index not in by_index:
            raise ValueError(f"Unsupported target_index from verifier: {target_index}")
        if target_index in out_by_index:
            raise ValueError(f"Duplicate target_index from verifier: {target_index}")
        target = by_index[target_index]
        if entity_type != target["type"]:
            raise ValueError(f"Verifier changed type for target_index {target_index}.")
        if not isinstance(canonical, str):
            raise ValueError(f"Verifier returned non-string canonical for target_index {target_index}.")
        if not isinstance(present, bool):
            raise ValueError(f"Verifier returned non-boolean present for target_index {target_index}.")
        if not isinstance(evidence, str):
            raise ValueError(f"Verifier returned non-string evidence for target_index {target_index}.")
        if present and not evidence.strip():
            raise ValueError(f"Verifier returned present=true without evidence for target_index {target_index}.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Verifier returned empty reason for target_index {target_index}.")
        out_by_index[target_index] = {
            "target_index": target_index,
            "type": target["type"],
            "canonical": target["canonical"],
            "present": present,
            "evidence": evidence,
            "reason": reason,
        }

    missing = [target["target_index"] for target in targets if target["target_index"] not in out_by_index]
    if missing:
        raise ValueError(f"Verifier omitted target_index values: {missing}")
    return [out_by_index[index] for index in range(len(targets))]


def stable_digest(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
