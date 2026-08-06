from __future__ import annotations

from pathlib import Path

import pytest

from voice_code_bench.figures import MODEL_LABELS, PAPER_MODEL_IDS, default_output_path, parse_overall_results, spearman


DATASET_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = DATASET_ROOT / "baselines" / "results.csv"


def test_paper_model_set_remains_frozen() -> None:
    rows = parse_overall_results(RESULTS_PATH, model_set="paper")
    assert len(rows) == len(PAPER_MODEL_IDS) == 12
    assert all(row["model"] != "Thinking Machines Inkling-NVFP4 via Modal (effort=max)" for row in rows)
    assert all(row["model"] != "Meta OmniASR LLM Unlimited 7B v2 via Modal" for row in rows)
    wer = [float(row["wer"]) for row in rows]
    assert spearman(wer, [float(row["ctem"]) for row in rows]) == pytest.approx(0.1678321678)
    assert spearman(wer, [float(row["tsr"]) for row in rows]) == pytest.approx(0.1996500435)


def test_all_model_set_includes_current_post_publication_results() -> None:
    assert MODEL_LABELS["modal_meta_omniasr_llm_unlimited_7b_v2"] == (
        "Meta OmniASR LLM Unlimited 7B v2 via Modal",
        "Batch",
    )
    assert MODEL_LABELS["inworld_stt_1"] == ("Inworld STT 1", "Batch")
    assert MODEL_LABELS["inworld_stt_1_streaming"] == ("Inworld STT 1", "Streaming")
    assert "inworld_stt_1" not in PAPER_MODEL_IDS
    assert "inworld_stt_1_streaming" not in PAPER_MODEL_IDS
    rows = parse_overall_results(RESULTS_PATH, model_set="all")
    assert len(rows) == 17
    inkling = next(
        row
        for row in rows
        if row["model"] == "Thinking Machines Inkling-NVFP4 via Modal (effort=max)"
    )
    assert inkling["wer"] == pytest.approx(4.8600918517)
    assert inkling["ctem"] == pytest.approx(84.2780026991)
    assert inkling["tsr"] == pytest.approx(49.6666666667)
    parakeet = next(
        row
        for row in rows
        if row["model"] == "NVIDIA Parakeet TDT 0.6B v3 via Modal"
    )
    assert parakeet["wer"] == pytest.approx(9.3177406373)
    assert parakeet["ctem"] == pytest.approx(78.6099865047)
    assert parakeet["tsr"] == pytest.approx(37.3333333333)
    omniasr = next(
        row
        for row in rows
        if row["model"] == "Meta OmniASR LLM Unlimited 7B v2 via Modal"
    )
    assert omniasr["wer"] == pytest.approx(10.5203352114)
    assert omniasr["ctem"] == pytest.approx(72.4696356275)
    assert omniasr["tsr"] == pytest.approx(25.6666666667)
    inworld_batch = next(row for row in rows if row["model"] == "Inworld STT 1" and row["mode"] == "Batch")
    assert inworld_batch["wer"] == pytest.approx(9.6728374603)
    assert inworld_batch["ctem"] == pytest.approx(83.7381916329)
    assert inworld_batch["tsr"] == pytest.approx(44.0)
    inworld_streaming = next(row for row in rows if row["model"] == "Inworld STT 1" and row["mode"] == "Streaming")
    assert inworld_streaming["wer"] == pytest.approx(57.3268311160)
    assert inworld_streaming["ctem"] == pytest.approx(33.8731443995)
    assert inworld_streaming["tsr"] == pytest.approx(6.3333333333)


def test_model_set_default_outputs_do_not_overlap() -> None:
    paper_output = DATASET_ROOT / "paper" / "figures" / "wer_entity_scatter.pdf"
    all_output = DATASET_ROOT / "baselines" / "figures" / "wer_entity_scatter.pdf"
    assert default_output_path(DATASET_ROOT, "paper") == paper_output
    assert default_output_path(DATASET_ROOT, "all") == all_output
    with pytest.raises(ValueError, match="Unknown figure model set"):
        default_output_path(DATASET_ROOT, "unknown")
