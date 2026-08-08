from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from voice_code_bench.commands.extract_entities import extract_grouped_models
from voice_code_bench.io import read_json_rows, write_json_rows
from voice_code_bench.verifier import load_verifier_config


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
            "id": f"{audio_id}_e01",
            "type": "code_symbol",
            "role": "value",
            "acoustic": canonical,
            "canonical": canonical,
        }],
        "entity_count": 1,
        "entity_types": ["code_symbol"],
    }


class GroupedSyncVerifierTests(unittest.TestCase):
    def test_groups_globally_checkpoints_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datapoints = [metadata_row(index) for index in range(11)]
            datapoint_by_id = {row["audio_id"]: row for row in datapoints}
            transcript_files: list[tuple[str, Path]] = []
            for model_id in ("model_one", "model_two"):
                path = root / model_id / "transcripts.json"
                write_json_rows(
                    path,
                    [
                        {"audio_id": row["audio_id"], "model_transcript": row["transcripts"]["canonical"]}
                        for row in datapoints
                    ],
                    run_metadata={"source": "test"},
                )
                transcript_files.append((model_id, path))

            calls: list[list[str]] = []
            active = 0
            max_active = 0
            lock = threading.Lock()

            def fake_verify(
                group_datapoints: list[dict[str, Any]],
                _transcript_rows: list[dict[str, Any]],
                *_args: Any,
            ) -> dict[str, list[dict[str, Any]]]:
                nonlocal active, max_active
                ids = [row["audio_id"] for row in group_datapoints]
                with lock:
                    calls.append(ids)
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {
                    row["audio_id"]: [{
                        "target_index": 0,
                        "type": "code_symbol",
                        "canonical": row["entities"][0]["canonical"],
                        "present": True,
                        "evidence": row["entities"][0]["canonical"],
                        "reason": "Exact value is present.",
                    }]
                    for row in group_datapoints
                }

            config = load_verifier_config("openai_gpt_5_6_sol_group10_sync_v1")
            output_dir = root / "output"
            with patch(
                "voice_code_bench.commands.extract_entities.verify_grouped_entity_matches",
                side_effect=fake_verify,
            ):
                paths = extract_grouped_models(
                    output_dir,
                    transcript_files,
                    datapoints,
                    datapoint_by_id,
                    {row["audio_id"]: index for index, row in enumerate(datapoints)},
                    config,
                    {},
                    concurrency=4,
                    resume=False,
                )

            self.assertEqual(sorted(len(group) for group in calls), [1, 1, 10, 10])
            self.assertEqual(max_active, 4)
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertEqual(len(read_json_rows(path)), 11)
                payload = json.loads(path.read_text())
                verifier_metadata = payload["run_metadata"]["entity_verifier"]
                self.assertEqual(verifier_metadata["datapoints_per_request"], 10)
                self.assertEqual(verifier_metadata["transport"], "sync")

            with patch(
                "voice_code_bench.commands.extract_entities.verify_grouped_entity_matches"
            ) as resumed_verify:
                extract_grouped_models(
                    output_dir,
                    transcript_files,
                    datapoints,
                    datapoint_by_id,
                    {row["audio_id"]: index for index, row in enumerate(datapoints)},
                    config,
                    {},
                    concurrency=4,
                    resume=True,
                )
            resumed_verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
