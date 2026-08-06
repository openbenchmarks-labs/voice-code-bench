from __future__ import annotations

import argparse
import importlib
import importlib.resources
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from voice_code_bench.commands import common
from voice_code_bench.commands.extract_entities import extract_one
from voice_code_bench.commands.score import run_score_entities, run_score_wer
from voice_code_bench.io import write_json_rows, write_metadata_jsonl


def sample_datapoints() -> list[dict[str, object]]:
    return [
        {"audio_id": "a1"},
        {"audio_id": "a2"},
    ]


class CliRefactorTests(unittest.TestCase):
    def test_cli_exports_public_entry_points(self) -> None:
        cli = importlib.import_module("voice_code_bench.cli")
        for name in [
            "batch_verify_main",
            "run_benchmark_main",
            "transcribe_main",
            "extract_entities_main",
            "score_entities_main",
            "score_wer_main",
            "run_transcribe",
            "run_extract_entities",
            "run_score_entities",
            "run_score_wer",
            "run_batch_verify",
        ]:
            self.assertTrue(callable(getattr(cli, name)))
        self.assertNotIn("transcribe_model", cli.__all__)

    def test_command_modules_expose_parser_main_and_runner(self) -> None:
        modules = {
            "voice_code_bench.commands.batch_verify": ["batch_verify_parser", "batch_verify_main", "run_batch_verify"],
            "voice_code_bench.commands.run": ["run_benchmark_parser", "run_benchmark_main", "run_full_benchmark"],
            "voice_code_bench.commands.transcribe": ["transcribe_parser", "transcribe_main", "run_transcribe"],
            "voice_code_bench.commands.extract_entities": [
                "extract_entities_parser",
                "extract_entities_main",
                "run_extract_entities",
            ],
            "voice_code_bench.commands.score": [
                "score_entities_parser",
                "score_entities_main",
                "run_score_entities",
                "score_wer_parser",
                "score_wer_main",
                "run_score_wer",
            ],
        }
        for module_name, names in modules.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                for name in names:
                    self.assertTrue(callable(getattr(module, name)))

    def test_console_script_targets_resolve(self) -> None:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        scripts = project_scripts(pyproject_path)
        for command_name, target in scripts.items():
            with self.subTest(command=command_name):
                module_name, function_name = target.split(":", 1)
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, function_name)))

    def test_verifier_config_is_packaged(self) -> None:
        config_path = importlib.resources.files("voice_code_bench").joinpath(
            "verifiers/openai_gpt_5_5_v1.json"
        )
        self.assertTrue(config_path.is_file())
        batch_config_path = importlib.resources.files("voice_code_bench").joinpath(
            "verifiers/openai_gpt_5_6_sol_batch10_v1.json"
        )
        self.assertTrue(batch_config_path.is_file())

        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        self.assertEqual(
            package_data(pyproject_path).get("voice_code_bench"),
            ["verifiers/*.json"],
        )

    def test_limit_helpers_validate_and_preserve_order(self) -> None:
        datapoints = sample_datapoints()
        self.assertEqual(common.apply_limit(datapoints, None), datapoints)
        self.assertEqual(common.apply_limit(datapoints, 1), [{"audio_id": "a1"}])
        self.assertEqual(common.metadata_order(datapoints), {"a1": 0, "a2": 1})
        self.assertEqual(common.resolve_concurrency(None, 2, "--model-concurrency"), 2)
        self.assertEqual(common.resolve_concurrency(3, 2, "--model-concurrency"), 3)
        with self.assertRaisesRegex(ValueError, "--limit must be positive"):
            common.apply_limit(datapoints, 0)
        with self.assertRaisesRegex(ValueError, "--model-concurrency must be positive"):
            common.resolve_concurrency(0, 2, "--model-concurrency")

    def test_read_complete_rows_rejects_missing_artifact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "transcripts.json"
            write_json_rows(artifact_path, [{"audio_id": "a1", "model_transcript": "hello"}])
            with self.assertRaisesRegex(RuntimeError, "first_missing=a2"):
                common.read_complete_rows(artifact_path, sample_datapoints(), artifact_name="transcript")

    def test_read_complete_rows_rejects_transcripts_without_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "transcripts.json"
            write_json_rows(artifact_path, [{"audio_id": "a1"}])
            with self.assertRaisesRegex(ValueError, "missing model_transcript"):
                common.read_complete_rows(artifact_path, [{"audio_id": "a1"}], artifact_name="transcript")


    def test_extract_one_preserves_transcript_metadata(self) -> None:
        datapoint = minimal_metadata_rows()[0]
        transcript_row = {
            "audio_id": "a1",
            "model_transcript": "call a at example dot com",
            "metadata": {"usage": {"modelId": "inworld/inworld-stt-1"}},
        }
        with mock.patch(
            "voice_code_bench.commands.extract_entities.verify_entity_matches",
            return_value=[
                {
                    "target_index": 0,
                    "type": "email_address",
                    "canonical": "a@example.com",
                    "present": True,
                    "evidence": "a at example dot com",
                    "reason": "The transcript contains the email.",
                }
            ],
        ):
            row = extract_one(datapoint, transcript_row, mock.MagicMock(), None, "live", {})
        self.assertEqual(row["metadata"], {"usage": {"modelId": "inworld/inworld-stt-1"}})

    def test_score_commands_run_on_minimal_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "data" / "metadata.jsonl"
            metadata_path.parent.mkdir(parents=True)
            write_metadata_jsonl(metadata_path, minimal_metadata_rows())
            model_dir = root / "predictions" / "toy_model"
            write_json_rows(
                model_dir / "transcripts.json",
                [{"audio_id": "a1", "model_transcript": "call a at example dot com"}],
            )
            write_json_rows(
                model_dir / "entity_matches.json",
                [
                    {
                        "audio_id": "a1",
                        "model_transcript": "call a at example dot com",
                        "entity_matches": [
                            {
                                "target_index": 0,
                                "type": "email_address",
                                "canonical": "a@example.com",
                                "present": True,
                                "evidence": "a at example dot com",
                                "reason": "The transcript contains the email.",
                            }
                        ],
                    }
                ],
            )
            output_csv = root / "results.csv"
            common_args = {
                "dataset_root": root,
                "metadata": metadata_path,
                "output_dir": root / "scores",
                "output_csv": output_csv,
                "scores_dir": None,
                "limit": None,
            }

            run_score_entities(argparse.Namespace(entity_matches_dir=root / "predictions", **common_args))
            run_score_wer(argparse.Namespace(transcripts_dir=root / "predictions", **common_args))

            results = output_csv.read_text(encoding="utf-8")
            self.assertIn("toy_model", results)
            self.assertIn("TSR", results)
            self.assertIn("WER", results)
            self.assertNotIn("WER Reference", results.splitlines()[0])


def project_scripts(pyproject_path: Path) -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and line.startswith("["):
            break
        if in_scripts and line and not line.startswith("#"):
            name, target = line.split("=", 1)
            scripts[name.strip()] = target.strip().strip('"')
    return scripts


def minimal_metadata_rows() -> list[dict[str, object]]:
    return [
        {
            "audio_id": "a1",
            "file_name": "audio/001.wav",
            "duration": 1.0,
            "transcripts": {
                "template": "call {email}",
                "acoustic": "call a at example dot com",
                "canonical": "call a@example.com",
            },
            "entities": [
                {
                    "id": "a1_e01",
                    "type": "email_address",
                    "role": "contact",
                    "acoustic": "a at example dot com",
                    "canonical": "a@example.com",
                }
            ],
            "entity_count": 1,
            "entity_types": ["email_address"],
            "domain": "support",
            "scenario": "callback",
            "difficulty": "light",
        }
    ]


def package_data(pyproject_path: Path) -> dict[str, list[str]]:
    data: dict[str, list[str]] = {}
    in_package_data = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[tool.setuptools.package-data]":
            in_package_data = True
            continue
        if in_package_data and line.startswith("["):
            break
        if in_package_data and line and not line.startswith("#"):
            name, value = line.split("=", 1)
            items = [
                item.strip().strip('"')
                for item in value.strip().strip("[]").split(",")
                if item.strip()
            ]
            data[name.strip()] = items
    return data


if __name__ == "__main__":
    unittest.main()
