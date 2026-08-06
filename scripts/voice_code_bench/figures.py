from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SHORT_MODEL_LABELS = {
    "Amazon Transcribe": "Amazon\nTranscribe",
    "AssemblyAI Universal-3 Pro": "AssemblyAI\nUniversal-3\nPro",
    "Deepgram Nova-3": "Deepgram\nNova-3",
    "ElevenLabs Scribe v2": "ElevenLabs\nScribe v2",
    "Inworld STT 1": "Inworld\nSTT 1",
    "Thinking Machines Inkling-NVFP4 via Modal (effort=max)": "Inkling-NVFP4\nvia Modal\n(effort=max)",
    "NVIDIA Parakeet TDT 0.6B v3 via Modal": "NVIDIA Parakeet\nTDT 0.6B v3\nvia Modal",
    "Meta OmniASR LLM Unlimited 7B v2 via Modal": "Meta OmniASR\nUnlimited 7B v2\nvia Modal",
}

MODEL_LABELS = {
    "amazon_transcribe_streaming": ("Amazon Transcribe", "Streaming"),
    "assemblyai_universal_3_pro": ("AssemblyAI Universal-3 Pro", "Batch"),
    "assemblyai_universal_3_pro_streaming": ("AssemblyAI Universal-3 Pro", "Streaming"),
    "deepgram_nova3": ("Deepgram Nova-3", "Batch"),
    "deepgram_nova3_streaming": ("Deepgram Nova-3", "Streaming"),
    "elevenlabs_scribe_v2": ("ElevenLabs Scribe v2", "Batch"),
    "elevenlabs_scribe_v2_realtime_streaming": ("ElevenLabs Scribe v2 realtime", "Streaming"),
    "google_cloud_chirp_3": ("Google Cloud Chirp 3", "Batch"),
    "google_cloud_chirp_3_streaming": ("Google Cloud Chirp 3", "Streaming"),
    "inworld_stt_1": ("Inworld STT 1", "Batch"),
    "inworld_stt_1_streaming": ("Inworld STT 1", "Streaming"),
    "modal_inkling": ("Thinking Machines Inkling-NVFP4 via Modal (effort=max)", "Batch"),
    "modal_meta_omniasr_llm_unlimited_7b_v2": ("Meta OmniASR LLM Unlimited 7B v2 via Modal", "Batch"),
    "modal_nvidia_parakeet_tdt_0_6b_v3": ("NVIDIA Parakeet TDT 0.6B v3 via Modal", "Batch"),
    "openai_gpt_4o_transcribe": ("OpenAI gpt-4o-transcribe", "Batch"),
    "openai_gpt_realtime_whisper_streaming": ("OpenAI gpt-realtime-whisper", "Streaming"),
    "whisper_large_v3": ("Whisper large-v3", "Batch"),
}

PAPER_MODEL_IDS = frozenset(
    {
        "amazon_transcribe_streaming",
        "assemblyai_universal_3_pro",
        "assemblyai_universal_3_pro_streaming",
        "deepgram_nova3",
        "deepgram_nova3_streaming",
        "elevenlabs_scribe_v2",
        "elevenlabs_scribe_v2_realtime_streaming",
        "google_cloud_chirp_3",
        "google_cloud_chirp_3_streaming",
        "openai_gpt_4o_transcribe",
        "openai_gpt_realtime_whisper_streaming",
        "whisper_large_v3",
    }
)
MODEL_SETS = ("paper", "all")

EXTRA_PANEL_LABELS = {
    ("tsr", "ElevenLabs Scribe v2", "Batch"): "ElevenLabs Scribe v2 callout",
}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark figures from baseline result artifacts.")
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=None, help="Defaults to baselines/results.csv under --dataset-root.")
    parser.add_argument(
        "--model-set",
        choices=MODEL_SETS,
        default="paper",
        help="Use the frozen paper baselines or all current baseline rows (default: paper).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to the paper figure for --model-set paper and the baselines figure for --model-set all.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    results_path = args.results or dataset_root / "baselines" / "results.csv"
    output_path = args.output or default_output_path(dataset_root, args.model_set)
    write_wer_entity_scatter(results_path, output_path, dataset_root=dataset_root, model_set=args.model_set)


def default_output_path(dataset_root: Path, model_set: str) -> Path:
    if model_set == "paper":
        return dataset_root / "paper" / "figures" / "wer_entity_scatter.pdf"
    if model_set == "all":
        return dataset_root / "baselines" / "figures" / "wer_entity_scatter.pdf"
    raise ValueError(f"Unknown figure model set: {model_set}")


def write_wer_entity_scatter(
    results_path: Path,
    output_path: Path,
    *,
    dataset_root: Path,
    model_set: str = "paper",
) -> None:
    import matplotlib.pyplot as plt

    rows = parse_overall_results(results_path, model_set=model_set)
    wer = [float(row["wer"]) for row in rows]
    ctem = [float(row["ctem"]) for row in rows]
    tsr = [float(row["tsr"]) for row in rows]
    ctem_rho = spearman(wer, ctem)
    tsr_rho = spearman(wer, tsr)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    scatter_panel(
        axes[0],
        rows,
        "ctem",
        "CTEM vs WER",
        "#0f5c99",
        "CTEM (%)",
        (72, 94),
        [72, 77, 82, 87, 92],
        True,
    )
    scatter_panel(
        axes[1],
        rows,
        "tsr",
        "TSR vs WER",
        "#b45309",
        "TSR (%)",
        (30, 70),
        [30, 35, 40, 45, 50, 55, 60, 65, 70],
        False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {display_path(output_path, dataset_root)}")
    print(f"WER vs CTEM Spearman rho: {ctem_rho:.3f}")
    print(f"WER vs TSR Spearman rho: {tsr_rho:.3f}")


def parse_overall_results(results_path: Path, *, model_set: str = "paper") -> list[dict[str, float | str]]:
    if model_set not in MODEL_SETS:
        raise ValueError(f"Unknown figure model set: {model_set}")
    rows: list[dict[str, float | str]] = []
    seen_model_ids: set[str] = set()
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model_id = str(row["Model"])
            if model_id in seen_model_ids:
                raise RuntimeError(f"Duplicate baseline result row: {model_id}")
            seen_model_ids.add(model_id)
            if model_set == "paper" and model_id not in PAPER_MODEL_IDS:
                continue
            try:
                model, mode = MODEL_LABELS[model_id]
            except KeyError as exc:
                raise RuntimeError(f"Missing figure label for baseline model: {model_id}") from exc
            wer = float(row["WER"]) * 100
            ctem = float(row["CTEM"]) * 100
            tsr = float(row["TSR"]) * 100
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "wer": wer,
                    "ctem": ctem,
                    "tsr": tsr,
                }
            )
    rows.sort(key=lambda row: float(row["ctem"]), reverse=True)
    if model_set == "paper":
        missing_model_ids = sorted(PAPER_MODEL_IDS - seen_model_ids)
        if missing_model_ids:
            missing = ", ".join(missing_model_ids)
            raise RuntimeError(f"Missing frozen paper baseline rows: {missing}")
    if not rows:
        raise RuntimeError(f"No baseline rows found for figure model set: {model_set}")
    return rows


def display_model(row: dict[str, float | str]) -> str:
    model = str(row["model"])
    label = SHORT_MODEL_LABELS.get(model, model)
    if str(row["mode"]) == "Streaming" and "streaming" not in model.lower() and "realtime" not in model.lower():
        return f"{label}\nstreaming"
    return label


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(values):
        end = idx
        while end + 1 < len(values) and values[order[end + 1]] == values[order[idx]]:
            end += 1
        avg = (idx + 1 + end + 1) / 2
        for rank_idx in range(idx, end + 1):
            out[order[rank_idx]] = avg
        idx = end + 1
    return out


def spearman(left: list[float], right: list[float]) -> float:
    left_ranks = ranks(left)
    right_ranks = ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((left_rank - left_mean) ** 2 for left_rank in left_ranks)
        * sum((right_rank - right_mean) ** 2 for right_rank in right_ranks)
    )
    return numerator / denominator


def extrema_labels(rows: list[dict[str, float | str]], metric: str) -> dict[int, list[str]]:
    metric_min = min(float(row[metric]) for row in rows)
    metric_max = max(float(row[metric]) for row in rows)
    wer_min = min(float(row["wer"]) for row in rows)
    wer_max = max(float(row["wer"]) for row in rows)

    def top_right_score(row: dict[str, float | str]) -> float:
        metric_position = (float(row[metric]) - metric_min) / (metric_max - metric_min)
        wer_position = (float(row["wer"]) - wer_min) / (wer_max - wer_min)
        return metric_position + wer_position

    top_right_row = max(rows, key=top_right_score)
    extrema = [
        (max(rows, key=lambda row: float(row[metric])), f"Highest {metric.upper()}"),
        (min(rows, key=lambda row: float(row[metric])), f"Lowest {metric.upper()}"),
        (min(rows, key=lambda row: float(row["wer"])), "Lowest WER"),
        (max(rows, key=lambda row: float(row["wer"])), "Highest WER"),
        (top_right_row, "Top right"),
    ]
    labels: dict[int, list[str]] = {}
    for row, label in extrema:
        labels.setdefault(id(row), []).append(label)
    return labels


def label_offset(metric: str, extrema: list[str]) -> tuple[tuple[int, int], str, str]:
    if metric == "ctem":
        if "Top right" in extrema:
            return (12, 11), "left", "bottom"
        if "Highest CTEM" in extrema:
            return (11, 0), "left", "center"
        if "Lowest WER" in extrema:
            return (-10, 21), "right", "bottom"
        return (14, -18), "left", "top"

    if "Top right" in extrema:
        return (11, 8), "left", "bottom"
    if "ElevenLabs Scribe v2 callout" in extrema:
        return (-10, -8), "right", "top"
    if "Highest TSR" in extrema:
        return (-10, 8), "right", "bottom"
    if "Lowest WER" in extrema:
        return (-10, 13), "right", "bottom"
    if "Lowest TSR" in extrema:
        return (14, -34), "left", "top"
    return (14, -10), "left", "top"


def scatter_panel(
    ax: object,
    rows: list[dict[str, float | str]],
    metric: str,
    title: str,
    color: str,
    x_label: str,
    x_limits: tuple[float, float],
    x_ticks: list[float],
    show_y_axis: bool,
) -> None:
    wer = [float(row["wer"]) for row in rows]
    metric_values = [float(row[metric]) for row in rows]
    ax.scatter(
        metric_values,
        wer,
        s=54,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=9)
    ax.set_xlabel(x_label, fontsize=8)
    metric_min = min(metric_values)
    metric_max = max(metric_values)
    if metric_min < x_limits[0] or metric_max > x_limits[1]:
        metric_span = metric_max - metric_min
        metric_padding = max(1.0, metric_span * 0.05)
        ax.set_xlim(metric_min - metric_padding, metric_max + metric_padding)
    else:
        ax.set_xlim(*x_limits)
        ax.set_xticks(x_ticks)
    wer_span = max(wer) - min(wer)
    wer_padding = max(0.5, wer_span * 0.05)
    ax.set_ylim(max(0.0, min(wer) - wer_padding), max(wer) + wer_padding)
    if show_y_axis:
        ax.set_ylabel("WER (%)", fontsize=8)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
    ax.tick_params(axis="both", labelsize=7, length=3, width=0.7, colors="#374151")
    ax.grid(True, color="#e5e7eb", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#6b7280")
        ax.spines[spine].set_linewidth(0.8)

    labels_by_row_id = extrema_labels(rows, metric)
    for row in rows:
        reason = EXTRA_PANEL_LABELS.get((metric, str(row["model"]), str(row["mode"])))
        if reason:
            labels_by_row_id.setdefault(id(row), []).append(reason)

    for row in rows:
        extrema = labels_by_row_id.get(id(row))
        if not extrema:
            continue
        x = float(row[metric])
        y = float(row["wer"])
        label = display_model(row)
        xytext, horizontal_alignment, vertical_alignment = label_offset(metric, extrema)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=6.4,
            ha=horizontal_alignment,
            va=vertical_alignment,
            color="#111827",
            linespacing=1.05,
            arrowprops={
                "arrowstyle": "-",
                "color": "#6b7280",
                "linewidth": 0.45,
                "shrinkA": 1,
                "shrinkB": 4,
            },
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=4,
        )


def display_path(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(dataset_root.resolve()))
    except ValueError:
        return str(path)
