from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from voice_code_bench.batch_verifier import (
    MAX_BATCH_ATTEMPTS,
    batch_state_dir,
    build_grouped_request,
    load_manifest,
    parse_grouped_response,
    plan_batches,
    reconcile_batches,
    validate_manifest,
)
from voice_code_bench.io import write_json_rows, write_metadata_jsonl
from voice_code_bench.http import HttpRequestError
from voice_code_bench.verifier import (
    VerifierCache,
    build_cache_entry,
    load_verifier_config,
    verifier_cache_key,
    verifier_input,
)


def metadata_row(index: int) -> dict[str, Any]:
    audio_id = f"a{index:02d}"
    canonical = f"value{index}"
    return {
        "audio_id": audio_id,
        "file_name": f"audio/{audio_id}.wav",
        "duration": 1.0,
        "domain": "software",
        "scenario": "test",
        "difficulty": "light",
        "transcripts": {"template": canonical, "acoustic": canonical, "canonical": canonical},
        "entities": [{
            "id": f"{audio_id}_e01", "type": "code_symbol", "role": "value",
            "acoustic": canonical, "canonical": canonical,
        }],
        "entity_count": 1,
        "entity_types": ["code_symbol"],
    }


def entity_result(index: int) -> dict[str, Any]:
    return {
        "target_index": 0, "type": "code_symbol", "canonical": f"value{index}",
        "present": True, "evidence": f"value{index}", "reason": "Exact value is present.",
    }


def grouped_payload(ids: list[str]) -> dict[str, Any]:
    return {"datapoints": [
        {"datapoint_id": audio_id, "entities": [entity_result(int(audio_id[1:]))]}
        for audio_id in ids
    ]}


class FakeBatchClient:
    def __init__(self, *, omit_once: str | None = None):
        self.upload_count = 0
        self.submit_count = 0
        self.batches: dict[str, dict[str, Any]] = {}
        self.files: dict[str, bytes] = {}
        self.uploaded_lines: dict[str, list[dict[str, Any]]] = {}
        self.omit_once = omit_once

    def upload(self, path: Path) -> dict[str, Any]:
        self.upload_count += 1
        file_id = f"file-in-{self.upload_count}"
        self.uploaded_lines[file_id] = [json.loads(line) for line in path.read_text().splitlines()]
        return {"id": file_id}

    def find_by_submission_key(self, submission_key: str) -> dict[str, Any] | None:
        return next((batch for batch in self.batches.values() if batch["metadata"]["vcb_submission_key"] == submission_key), None)

    def submit(self, input_file_id: str, completion_window: str, metadata: dict[str, str]) -> dict[str, Any]:
        self.submit_count += 1
        batch_id = f"batch-{self.submit_count}"
        output_id = f"file-out-{self.submit_count}"
        output = []
        for line in self.uploaded_lines[input_file_id]:
            request = json.loads(line["body"]["messages"][1]["content"])
            ids = [item["datapoint_id"] for item in request["datapoints"]]
            if self.omit_once in ids:
                ids.remove(self.omit_once)
                self.omit_once = None
            output.append({
                "custom_id": line["custom_id"],
                "response": {"status_code": 200, "body": {
                    "choices": [{"message": {"content": json.dumps(grouped_payload(ids))}}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }},
            })
        self.files[output_id] = "".join(json.dumps(row) + "\n" for row in reversed(output)).encode()
        batch = {"id": batch_id, "status": "completed", "output_file_id": output_id, "error_file_id": None, "metadata": metadata}
        self.batches[batch_id] = batch
        return batch

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return self.batches[batch_id]

    def download(self, file_id: str) -> bytes:
        return self.files[file_id]


class PendingBatchClient(FakeBatchClient):
    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return {**self.batches[batch_id], "status": "in_progress", "output_file_id": None}


class RateLimitedBatchClient(FakeBatchClient):
    def submit(self, input_file_id: str, completion_window: str, metadata: dict[str, str]) -> dict[str, Any]:
        raise HttpRequestError("queue full", status_code=429)


class MalformedBatchClient(FakeBatchClient):
    def submit(self, input_file_id: str, completion_window: str, metadata: dict[str, str]) -> dict[str, Any]:
        result = super().submit(input_file_id, completion_window, metadata)
        self.files[result["output_file_id"]] = b'{"custom_id":"wrong","response":{"status_code":500}}\n'
        return result


class BatchVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_verifier_config("openai_gpt_5_6_sol_batch10_v1")

    def make_run(self, root: Path, count: int) -> tuple[Path, Path]:
        metadata = root / "data" / "metadata.jsonl"
        metadata.parent.mkdir(parents=True)
        rows = [metadata_row(index) for index in range(count)]
        write_metadata_jsonl(metadata, rows)
        run_dir = root / "runs" / "full"
        write_json_rows(
            run_dir / "toy" / "transcripts.json",
            [{"audio_id": row["audio_id"], "model_transcript": row["transcripts"]["canonical"]} for row in rows],
            run_metadata={"source": "test"},
        )
        return run_dir, metadata

    def add_model(self, run_dir: Path, count: int, model_id: str) -> None:
        write_json_rows(
            run_dir / model_id / "transcripts.json",
            [{"audio_id": f"a{index:02d}", "model_transcript": f"value{index}"} for index in range(count)],
        )

    def test_plan_groups_ten_then_one_and_versions_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 11)
            manifest = plan_batches(run_dir, root, metadata, self.config)
            attempt = manifest["models"]["toy"]["attempts"][0]
            self.assertEqual([len(group["datapoint_ids"]) for group in attempt["groups"]], [10, 1])
            lines = [json.loads(line) for line in (batch_state_dir(run_dir, self.config.id) / attempt["request_file"]).read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["method"], "POST")
            self.assertEqual(lines[0]["url"], "/v1/chat/completions")
            self.assertEqual(lines[0]["body"]["model"], "gpt-5.6-sol")
            self.assertEqual(lines[0]["body"]["reasoning_effort"], "medium")
            self.assertTrue(lines[0]["custom_id"].startswith("vcb-01-0000-"))

    def test_replan_is_idempotent_and_rejects_changed_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 1)
            first = plan_batches(run_dir, root, metadata, self.config)
            second = plan_batches(run_dir, root, metadata, self.config)
            self.assertEqual(first, second)
            request_path = batch_state_dir(run_dir, self.config.id) / first["models"]["toy"]["attempts"][0]["request_file"]
            audit_before = request_path.read_bytes()
            write_json_rows(run_dir / "toy" / "transcripts.json", [{"audio_id": "a00", "model_transcript": "changed"}])
            with self.assertRaisesRegex(RuntimeError, "different inputs"):
                plan_batches(run_dir, root, metadata, self.config)
            self.assertEqual(request_path.read_bytes(), audit_before)

    def test_grouped_parser_reorders_and_rejects_id_corruption(self) -> None:
        datapoints = {f"a{index:02d}": metadata_row(index) for index in range(2)}
        parsed = parse_grouped_response(grouped_payload(["a01", "a00"]), ["a00", "a01"], datapoints)
        self.assertEqual(list(parsed), ["a01", "a00"])
        bad_cases = [
            ({"datapoints": []}, "omitted"),
            ({"datapoints": [grouped_payload(["a00"])["datapoints"][0]] * 2}, "Duplicate"),
            (grouped_payload(["a99"]), "Unknown"),
            ({"datapoints": "bad"}, "datapoints array"),
        ]
        for payload, message in bad_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_grouped_response(payload, ["a00", "a01"], datapoints)

    def test_end_to_end_collects_shuffled_output_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 11)
            plan_batches(run_dir, root, metadata, self.config)
            client = FakeBatchClient()
            manifest, _ = reconcile_batches(run_dir, self.config, {}, client=client)
            self.assertEqual(manifest["models"]["toy"]["state"], "collected")
            self.assertEqual(client.submit_count, 1)
            output = json.loads((run_dir / "toy" / "entity_matches.json").read_text())
            self.assertEqual([row["audio_id"] for row in output["items"]], [f"a{i:02d}" for i in range(11)])
            self.assertEqual(output["run_metadata"]["entity_verifier"]["datapoints_per_request"], 10)
            validate_manifest(run_dir, self.config)

    def test_invalid_group_retries_only_its_unresolved_datapoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 11)
            plan_batches(run_dir, root, metadata, self.config)
            client = FakeBatchClient(omit_once="a10")
            first, _ = reconcile_batches(run_dir, self.config, {}, client=client)
            model = first["models"]["toy"]
            self.assertEqual(model["state"], "built")
            self.assertEqual(model["unresolved_ids"], ["a10"])
            self.assertEqual(model["attempts"][1]["datapoint_count"], 1)
            second, _ = reconcile_batches(run_dir, self.config, {}, client=client)
            self.assertEqual(second["models"]["toy"]["state"], "collected")
            self.assertEqual(client.submit_count, 2)

    def test_submission_recovery_does_not_submit_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 1)
            plan_batches(run_dir, root, metadata, self.config)
            client = FakeBatchClient()
            manifest = load_manifest(run_dir, self.config.id)
            attempt = manifest["models"]["toy"]["attempts"][0]
            uploaded = client.upload(batch_state_dir(run_dir, self.config.id) / attempt["request_file"])
            remote = client.submit(uploaded["id"], "24h", {"vcb_submission_key": attempt["submission_key"]})
            attempt["input_file_id"] = uploaded["id"]
            attempt["state"] = "submitting"
            from voice_code_bench.batch_verifier import save_manifest
            save_manifest(run_dir, self.config.id, manifest)
            resumed, _ = reconcile_batches(run_dir, self.config, {}, client=client)
            self.assertEqual(resumed["models"]["toy"]["state"], "collected")
            self.assertEqual(client.submit_count, 1)
            self.assertEqual(remote["id"], resumed["models"]["toy"]["attempts"][0]["batch_id"])

    def test_batch_config_request_helper(self) -> None:
        request = build_grouped_request(self.config, [{"datapoint_id": "a", "target_entities": [], "transcript": "x"}])
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertIn('"datapoints"', request["messages"][1]["content"])
        self.assertEqual(MAX_BATCH_ATTEMPTS, 3)

    def test_in_flight_cap_defers_second_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 1)
            self.add_model(run_dir, 1, "toy_two")
            plan_batches(run_dir, root, metadata, self.config)
            client = PendingBatchClient()
            manifest, _ = reconcile_batches(run_dir, self.config, {}, client=client, max_in_flight=1)
            states = [model["state"] for model in manifest["models"].values()]
            self.assertEqual(states.count("submitted"), 1)
            self.assertEqual(client.submit_count, 1)

    def test_429_defers_without_consuming_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 1)
            plan_batches(run_dir, root, metadata, self.config)
            manifest, _ = reconcile_batches(run_dir, self.config, {}, client=RateLimitedBatchClient())
            model = manifest["models"]["toy"]
            self.assertEqual(model["state"], "uploaded")
            self.assertEqual(len(model["attempts"]), 1)
            self.assertIn("deferred_reason", model["attempts"][0])

    def test_malformed_outputs_stop_after_three_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 1)
            plan_batches(run_dir, root, metadata, self.config)
            client = MalformedBatchClient()
            manifest = None
            for _ in range(MAX_BATCH_ATTEMPTS):
                manifest, _ = reconcile_batches(run_dir, self.config, {}, client=client)
            assert manifest is not None
            model = manifest["models"]["toy"]
            self.assertEqual(model["state"], "failed")
            self.assertEqual(len(model["attempts"]), MAX_BATCH_ATTEMPTS)
            self.assertFalse((run_dir / "toy" / "entity_matches.json").exists())

    def test_replay_cache_collects_without_an_api_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, metadata = self.make_run(root, 1)
            cache_path = root / "cache.json"
            cache = VerifierCache(cache_path)
            datapoint = metadata_row(0)
            request_input = verifier_input(datapoint, "value0")
            response = {"entities": [entity_result(0)]}
            cache.put(build_cache_entry(self.config, request_input, verifier_cache_key(self.config, request_input), response))
            planned = plan_batches(run_dir, root, metadata, self.config, cache_path=cache_path, cache_mode="replay")
            self.assertEqual(planned["models"]["toy"]["unresolved_ids"], [])
            collected, _ = reconcile_batches(run_dir, self.config, {})
            self.assertEqual(collected["models"]["toy"]["state"], "collected")


if __name__ == "__main__":
    unittest.main()
