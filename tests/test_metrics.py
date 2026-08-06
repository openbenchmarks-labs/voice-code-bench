from __future__ import annotations

import pytest

from voice_code_bench.metrics import (
    FORMAT_INVARIANT_WER_REFERENCE,
    compute_format_invariant_wer,
    compute_wer,
    score_wer,
)


def phone_datapoint() -> dict[str, object]:
    return {
        "audio_id": "phone",
        "transcripts": {
            "template": "Call area code <phone_e01> tomorrow",
            "acoustic": "Call area code two one two five five five zero one zero zero tomorrow",
            "canonical": "Call area code (212) 555-0100 tomorrow",
        },
        "entities": [
            {
                "id": "phone_e01",
                "type": "phone_number",
                "role": "callback_number",
                "acoustic": "two one two five five five zero one zero zero",
                "canonical": "(212) 555-0100",
            }
        ],
    }


def test_format_invariant_wer_treats_entity_renderings_as_equivalent() -> None:
    datapoint = phone_datapoint()
    formatted = "Call area code 212-555-0100 tomorrow"

    assert compute_wer(datapoint["transcripts"]["acoustic"], formatted).wer > 0
    result = compute_format_invariant_wer(datapoint, formatted)
    acoustic_result = compute_format_invariant_wer(
        datapoint,
        datapoint["transcripts"]["acoustic"],
    )

    assert result.wer == 0
    assert acoustic_result.wer == 0
    assert result.reference_words == 14
    assert result.hypothesis_words == 7


def test_format_invariant_wer_still_counts_recognition_errors() -> None:
    datapoint = phone_datapoint()
    result = compute_format_invariant_wer(
        datapoint,
        "Call area code 212-555-0199 next week",
    )

    assert result.wer > 0
    assert result.substitutions + result.deletions + result.insertions > 0


def test_score_wer_uses_format_invariant_entity_normalization() -> None:
    datapoint = phone_datapoint()
    score = score_wer(
        datapoint,
        "test_model",
        {
            "audio_id": "phone",
            "model_transcript": "Call area code 212-555-0100 tomorrow",
        },
    )

    assert score["wer"]["wer"] == pytest.approx(0)
    assert score["wer_reference"] == FORMAT_INVARIANT_WER_REFERENCE
