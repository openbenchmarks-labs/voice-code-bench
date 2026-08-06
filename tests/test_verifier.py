from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from voice_code_bench.models import DEFAULT_VERIFIER_ID
from voice_code_bench import verifier
from voice_code_bench.verifier import (
    VerifierCache,
    build_openai_verify_request,
    load_verifier_config,
    parse_llm_entity_matches,
    verify_entity_matches,
)


def sample_datapoint() -> dict[str, Any]:
    return {
        "audio_id": "sample_001",
        "entities": [
            {
                "id": "sample_001_e01",
                "type": "email_address",
                "role": "contact",
                "acoustic": "a at example dot com",
                "canonical": "a@example.com",
            }
        ],
    }


def sample_response() -> dict[str, Any]:
    return {
        "entities": [
            {
                "target_index": 0,
                "type": "email_address",
                "canonical": "a@example.com",
                "present": True,
                "evidence": "a@example.com",
                "reason": "The transcript contains the email address.",
            }
        ]
    }


class VerifierTests(unittest.TestCase):
    def test_loads_default_versioned_verifier_config(self) -> None:
        config = load_verifier_config(DEFAULT_VERIFIER_ID)
        self.assertEqual(config.id, DEFAULT_VERIFIER_ID)
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-5.5")
        self.assertIn("Voice Code Bench", config.system_prompt)
        self.assertEqual(config.response_format["type"], "json_schema")
        self.assertEqual(len(config.digest), 64)

    def test_openai_request_comes_from_config_artifact(self) -> None:
        config = load_verifier_config(DEFAULT_VERIFIER_ID)
        request = build_openai_verify_request(config, sample_datapoint(), "send a@example.com")
        self.assertEqual(request["model"], "gpt-5.5")
        self.assertNotIn("temperature", request)
        self.assertEqual(request["response_format"], config.response_format)
        self.assertEqual(request["messages"][0]["content"], config.system_prompt)
        self.assertIn('"datapoint_id": "sample_001"', request["messages"][1]["content"])

    def test_openai_request_sets_temperature_for_non_gpt5_model_override(self) -> None:
        config = replace(load_verifier_config(DEFAULT_VERIFIER_ID), model="gpt-4.1")
        request = build_openai_verify_request(config, sample_datapoint(), "send a@example.com")
        self.assertEqual(request["temperature"], 0)

    def test_parse_llm_entity_matches_validates_required_invariants(self) -> None:
        self.assertEqual(
            parse_llm_entity_matches(sample_response(), sample_datapoint()["entities"]),
            [
                {
                    "target_index": 0,
                    "type": "email_address",
                    "canonical": "a@example.com",
                    "present": True,
                    "evidence": "a@example.com",
                    "reason": "The transcript contains the email address.",
                }
            ],
        )
        invalid_cases = [
            ({}, "entities array"),
            ({"entities": []}, "omitted target_index"),
            ({"entities": [{**sample_response()["entities"][0], "target_index": 1}]}, "Unsupported target_index"),
            ({"entities": [{**sample_response()["entities"][0], "type": "url"}]}, "changed type"),
            ({"entities": [{**sample_response()["entities"][0], "present": "yes"}]}, "non-boolean"),
            ({"entities": [{**sample_response()["entities"][0], "evidence": ""}]}, "without evidence"),
            ({"entities": [{**sample_response()["entities"][0], "reason": ""}]}, "empty reason"),
            (
                {"entities": [sample_response()["entities"][0], sample_response()["entities"][0]]},
                "Duplicate target_index",
            ),
        ]
        for payload, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_llm_entity_matches(payload, sample_datapoint()["entities"])

    def test_verifier_cache_live_replay_and_live_fill(self) -> None:
        config = load_verifier_config(DEFAULT_VERIFIER_ID)
        datapoint = sample_datapoint()
        transcript = "send a@example.com"
        calls: list[str] = []

        def fake_call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls.append("called")
            return sample_response()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VerifierCache(Path(tmpdir) / "verifier-cache.json")
            original_call = verifier.call_verifier_provider
            verifier.call_verifier_provider = fake_call
            try:
                with self.assertRaisesRegex(RuntimeError, "cache miss"):
                    verify_entity_matches(datapoint, transcript, config, {}, cache=cache, cache_mode="replay")

                live_matches = verify_entity_matches(datapoint, transcript, config, {}, cache=cache, cache_mode="live")
                self.assertTrue(live_matches[0]["present"])
                self.assertEqual(calls, ["called"])

                verifier.call_verifier_provider = lambda *_args, **_kwargs: self.fail("replay should not call provider")
                replay_matches = verify_entity_matches(datapoint, transcript, config, {}, cache=cache, cache_mode="replay")
                self.assertEqual(replay_matches, live_matches)

                live_fill_matches = verify_entity_matches(datapoint, transcript, config, {}, cache=cache, cache_mode="live-fill")
                self.assertEqual(live_fill_matches, live_matches)
            finally:
                verifier.call_verifier_provider = original_call

    def test_verifier_cache_misses_when_config_digest_changes(self) -> None:
        config = load_verifier_config(DEFAULT_VERIFIER_ID)
        datapoint = sample_datapoint()
        transcript = "send a@example.com"

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VerifierCache(Path(tmpdir) / "verifier-cache.json")
            original_call = verifier.call_verifier_provider
            verifier.call_verifier_provider = lambda *_args, **_kwargs: sample_response()
            try:
                verify_entity_matches(datapoint, transcript, config, {}, cache=cache, cache_mode="live")
                changed_config = replace(config, model="gpt-4.1")
                with self.assertRaisesRegex(RuntimeError, "cache miss"):
                    verify_entity_matches(datapoint, transcript, changed_config, {}, cache=cache, cache_mode="replay")
            finally:
                verifier.call_verifier_provider = original_call


if __name__ == "__main__":
    unittest.main()
