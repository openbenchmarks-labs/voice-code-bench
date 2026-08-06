from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ENTITY_TYPES, entity_level_column, sort_entity_types

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
TEMPLATE_ENTITY_RE = re.compile(r"<([^>]+)>")
FORMAT_INVARIANT_WER_REFERENCE = "acoustic_or_canonical_entity_lattice"


@dataclass(frozen=True)
class WerResult:
    reference_words: int
    hypothesis_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float


def entity_summary(entity: dict[str, Any], *, target_index: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "target_index": target_index,
        "id": entity["id"],
        "type": entity["type"],
        "canonical": entity["canonical"],
    }
    if "role" in entity:
        out["role"] = entity["role"]
    return out


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def compute_wer(reference_text: str, hypothesis_text: str) -> WerResult:
    return compute_token_wer(tokenize_words(reference_text), tokenize_words(hypothesis_text))


def compute_format_invariant_wer(datapoint: dict[str, Any], hypothesis_text: str) -> WerResult:
    reference = tokenize_words(datapoint["transcripts"]["acoustic"])
    hypothesis = tokenize_words(hypothesis_text)
    chunks = format_invariant_reference_chunks(datapoint, reference)
    return compute_alternative_reference_wer(chunks, hypothesis, reference_word_count=len(reference))


def format_invariant_reference_chunks(
    datapoint: dict[str, Any],
    reference: list[str],
) -> list[list[list[str]]]:
    """Build ordered reference chunks with acoustic/canonical entity alternatives."""
    entities = datapoint.get("entities")
    transcripts = datapoint.get("transcripts")
    if not isinstance(entities, list):
        raise ValueError(f"Datapoint {datapoint.get('audio_id', '<unknown>')} has invalid entities.")
    if not isinstance(transcripts, dict) or not isinstance(transcripts.get("template"), str):
        raise ValueError(f"Datapoint {datapoint.get('audio_id', '<unknown>')} is missing its template transcript.")

    entities_by_id: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("WER entity is missing its ID.")
        if entity_id in entities_by_id:
            raise ValueError(f"WER entities contain duplicate ID {entity_id}.")
        entities_by_id[entity_id] = entity
    entity_ids = TEMPLATE_ENTITY_RE.findall(transcripts["template"])
    if len(entity_ids) != len(entities) or set(entity_ids) != set(entities_by_id):
        entity_ids = list(entities_by_id)

    chunks: list[list[list[str]]] = []
    cursor = 0
    for entity_id in entity_ids:
        entity = entities_by_id[entity_id]
        acoustic = entity.get("acoustic")
        canonical = entity.get("canonical")
        if not isinstance(acoustic, str) or not isinstance(canonical, str):
            raise ValueError(f"Entity {entity_id} is missing acoustic or canonical text.")
        acoustic_tokens = tokenize_words(acoustic)
        canonical_tokens = tokenize_words(canonical)
        if not acoustic_tokens or not canonical_tokens:
            raise ValueError(f"Entity {entity_id} has an empty acoustic or canonical WER rendering.")
        start = find_subsequence(reference, acoustic_tokens, cursor)
        if start is None:
            raise ValueError(
                f"Datapoint {datapoint.get('audio_id', '<unknown>')} acoustic transcript "
                f"does not contain entity {entity_id} after token {cursor}."
            )
        if start > cursor:
            chunks.append([reference[cursor:start]])
        alternatives = [acoustic_tokens]
        if canonical_tokens != acoustic_tokens:
            alternatives.append(canonical_tokens)
        chunks.append(alternatives)
        cursor = start + len(acoustic_tokens)
    if cursor < len(reference):
        chunks.append([reference[cursor:]])
    return chunks


def find_subsequence(tokens: list[str], subsequence: list[str], start: int = 0) -> int | None:
    width = len(subsequence)
    for index in range(start, len(tokens) - width + 1):
        if tokens[index : index + width] == subsequence:
            return index
    return None


def compute_token_wer(reference: list[str], hypothesis: list[str]) -> WerResult:
    return compute_alternative_reference_wer([[reference]], hypothesis, reference_word_count=len(reference))


def compute_alternative_reference_wer(
    reference_chunks: list[list[list[str]]],
    hypothesis: list[str],
    *,
    reference_word_count: int,
) -> WerResult:
    """Find minimum edit distance through an acyclic alternative-reference lattice."""
    cols = len(hypothesis) + 1
    prev_cost = list(range(cols))
    prev_sub = [0] * cols
    prev_del = [0] * cols
    prev_ins = list(range(cols))

    for alternatives in reference_chunks:
        candidates = [
            advance_wer_state(
                prev_cost,
                prev_sub,
                prev_del,
                prev_ins,
                alternative,
                hypothesis,
            )
            for alternative in alternatives
        ]
        prev_cost = list(candidates[0][0])
        prev_sub = list(candidates[0][1])
        prev_del = list(candidates[0][2])
        prev_ins = list(candidates[0][3])
        for cost, substitutions, deletions, insertions in candidates[1:]:
            for col in range(cols):
                if cost[col] < prev_cost[col]:
                    prev_cost[col] = cost[col]
                    prev_sub[col] = substitutions[col]
                    prev_del[col] = deletions[col]
                    prev_ins[col] = insertions[col]

    substitutions = prev_sub[-1]
    deletions = prev_del[-1]
    insertions = prev_ins[-1]
    return WerResult(
        reference_words=reference_word_count,
        hypothesis_words=len(hypothesis),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        wer=0.0
        if reference_word_count == 0
        else (substitutions + deletions + insertions) / reference_word_count,
    )


def advance_wer_state(
    initial_cost: list[int],
    initial_sub: list[int],
    initial_del: list[int],
    initial_ins: list[int],
    reference: list[str],
    hypothesis: list[str],
) -> tuple[list[int], list[int], list[int], list[int]]:
    cols = len(hypothesis) + 1
    prev_cost = list(initial_cost)
    prev_sub = list(initial_sub)
    prev_del = list(initial_del)
    prev_ins = list(initial_ins)

    for reference_word in reference:
        curr_cost = [prev_cost[0] + 1] + [0] * (cols - 1)
        curr_sub = [0] * cols
        curr_del = [prev_del[0] + 1] + [0] * (cols - 1)
        curr_ins = [0] * cols

        for col, hypothesis_word in enumerate(hypothesis, start=1):
            substitution_increment = 0 if reference_word == hypothesis_word else 1

            best_cost = prev_cost[col - 1] + substitution_increment
            best_sub = prev_sub[col - 1] + substitution_increment
            best_del = prev_del[col - 1]
            best_ins = prev_ins[col - 1]

            deletion_cost = prev_cost[col] + 1
            if deletion_cost < best_cost:
                best_cost = deletion_cost
                best_sub = prev_sub[col]
                best_del = prev_del[col] + 1
                best_ins = prev_ins[col]

            insertion_cost = curr_cost[col - 1] + 1
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_sub = curr_sub[col - 1]
                best_del = curr_del[col - 1]
                best_ins = curr_ins[col - 1] + 1

            curr_cost[col] = best_cost
            curr_sub[col] = best_sub
            curr_del[col] = best_del
            curr_ins[col] = best_ins

        prev_cost = curr_cost
        prev_sub = curr_sub
        prev_del = curr_del
        prev_ins = curr_ins
    return prev_cost, prev_sub, prev_del, prev_ins


def score_entity_capture(datapoint: dict[str, Any], model_id: str, prediction: dict[str, Any]) -> dict[str, Any]:
    if prediction["audio_id"] != datapoint["audio_id"]:
        raise ValueError(f"Prediction audio_id {prediction['audio_id']} does not match {datapoint['audio_id']}.")
    if not isinstance(prediction.get("entity_matches"), list):
        raise ValueError("Prediction is missing entity_matches.")
    match_by_index = {}
    for match in prediction["entity_matches"]:
        target_index = match.get("target_index")
        if not isinstance(target_index, int):
            raise ValueError(f"Prediction entity_match is missing integer target_index: {match}")
        if target_index < 0 or target_index >= len(datapoint["entities"]):
            raise ValueError(f"Prediction entity_match has out-of-range target_index={target_index}.")
        if target_index in match_by_index:
            raise ValueError(f"Prediction has duplicate entity_match target_index={target_index}.")
        match_by_index[target_index] = match
    correct_entities = []
    missed_entities = []
    score_entity_matches = []
    for index, entity in enumerate(datapoint["entities"]):
        if index not in match_by_index:
            raise ValueError(
                f"Prediction entity_matches omitted target_index={index} for audio_id={prediction['audio_id']}."
            )
        match = match_by_index[index]
        if match.get("type") != entity["type"] or match.get("canonical") != entity["canonical"]:
            raise ValueError(
                f"Prediction entity_match target_index={index} changed the target entity "
                f"from {(entity['type'], entity['canonical'])} to {(match.get('type'), match.get('canonical'))}."
            )
        present = match.get("present")
        if not isinstance(present, bool):
            raise ValueError(f"Prediction entity_match target_index={index} has non-boolean present.")
        summary = entity_summary(entity, target_index=index)
        if present:
            correct_entities.append(summary)
        else:
            missed_entities.append(summary)
        score_entity_matches.append(
            {
                "target_index": index,
                "id": entity["id"],
                "type": entity["type"],
                "role": entity["role"],
                "canonical": entity["canonical"],
                "present": present,
                "evidence": str(match.get("evidence", "")),
                "reason": str(match.get("reason", "")),
            }
        )
    entity_types = sorted({str(entity["type"]) for entity in datapoint["entities"]})
    return {
        "datapoint_id": datapoint["audio_id"],
        "stt_model_id": model_id,
        "task_success": 1 if not missed_entities else 0,
        "gold_token_count": len(datapoint["entities"]),
        "correct_token_count": len(correct_entities),
        "correct": correct_entities,
        "missed": missed_entities,
        "entity_matches": score_entity_matches,
        "by_type": {
            entity_type: {
                "total": sum(1 for entity in datapoint["entities"] if entity["type"] == entity_type),
                "correct": sum(1 for entity in correct_entities if entity["type"] == entity_type),
            }
            for entity_type in entity_types
        },
    }


def aggregate_entity_scores(scores: list[dict[str, Any]], model_ids: list[str], datapoint_count: int) -> list[dict[str, Any]]:
    entity_types = sort_entity_types(
        [entity_type for score in scores for entity_type in score["by_type"].keys()]
    )
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_scores = [score for score in scores if score["stt_model_id"] == model_id]
        if len(model_scores) != datapoint_count:
            continue
        gold_tokens = sum(score["gold_token_count"] for score in model_scores)
        correct_tokens = sum(score["correct_token_count"] for score in model_scores)
        row: dict[str, Any] = {
            "Model": model_id,
            "TSR": divide(sum(score["task_success"] for score in model_scores), len(model_scores)),
            "CTEM": divide(correct_tokens, gold_tokens),
        }
        for entity_type in entity_types:
            total = sum(score["by_type"].get(entity_type, {}).get("total", 0) for score in model_scores)
            correct = sum(score["by_type"].get(entity_type, {}).get("correct", 0) for score in model_scores)
            row[entity_level_column(entity_type)] = divide(correct, total)
        rows.append(row)
    return rows


def aggregate_entity_match_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    total_entities = 0
    present_entities = 0
    task_success_count = 0
    for row in rows:
        matches = row.get("entity_matches", [])
        if not isinstance(matches, list):
            continue
        row_success = True
        for match in matches:
            if not isinstance(match, dict):
                continue
            entity_type = str(match.get("type", "unknown"))
            present = match.get("present") is True
            by_type.setdefault(entity_type, {"total": 0, "present": 0})
            by_type[entity_type]["total"] += 1
            total_entities += 1
            if present:
                by_type[entity_type]["present"] += 1
                present_entities += 1
            else:
                row_success = False
        if row_success:
            task_success_count += 1
    ordered_by_type = {
        entity_type: {
            "total": values["total"],
            "present": values["present"],
            "rate": divide(values["present"], values["total"]),
        }
        for entity_type, values in sorted(by_type.items(), key=lambda item: entity_type_sort_key(item[0]))
    }
    return {
        "datapoint_count": len(rows),
        "task_success_count": task_success_count,
        "task_success_rate": divide(task_success_count, len(rows)),
        "entity_count": total_entities,
        "present_entity_count": present_entities,
        "entity_capture_rate": divide(present_entities, total_entities),
        "by_type": ordered_by_type,
    }


def aggregate_entity_score_rows(scores: list[dict[str, Any]]) -> dict[str, Any]:
    gold_tokens = sum(score["gold_token_count"] for score in scores)
    correct_tokens = sum(score["correct_token_count"] for score in scores)
    by_type: dict[str, dict[str, int]] = {}
    for score in scores:
        for entity_type, values in score["by_type"].items():
            by_type.setdefault(entity_type, {"total": 0, "correct": 0})
            by_type[entity_type]["total"] += values.get("total", 0)
            by_type[entity_type]["correct"] += values.get("correct", 0)
    model_ids = sorted({str(score.get("stt_model_id", "")) for score in scores if score.get("stt_model_id")})
    return {
        "model_id": model_ids[0] if len(model_ids) == 1 else None,
        "datapoint_count": len(scores),
        "task_success_count": sum(score["task_success"] for score in scores),
        "task_success_rate": divide(sum(score["task_success"] for score in scores), len(scores)),
        "gold_token_count": gold_tokens,
        "correct_token_count": correct_tokens,
        "entity_capture_rate": divide(correct_tokens, gold_tokens),
        "by_type": {
            entity_type: {
                "total": values["total"],
                "correct": values["correct"],
                "rate": divide(values["correct"], values["total"]),
            }
            for entity_type, values in sorted(by_type.items(), key=lambda item: entity_type_sort_key(item[0]))
        },
    }


def score_wer(datapoint: dict[str, Any], model_id: str, transcript_row: dict[str, Any]) -> dict[str, Any]:
    if transcript_row["audio_id"] != datapoint["audio_id"]:
        raise ValueError(f"Transcript audio_id {transcript_row['audio_id']} does not match {datapoint['audio_id']}.")
    if not isinstance(transcript_row.get("model_transcript"), str):
        raise ValueError(f"Transcript row for {datapoint['audio_id']} is missing model_transcript.")
    wer = compute_format_invariant_wer(datapoint, transcript_row["model_transcript"])
    return {
        "datapoint_id": datapoint["audio_id"],
        "stt_model_id": model_id,
        "wer_reference": FORMAT_INVARIANT_WER_REFERENCE,
        "wer": wer.__dict__,
    }


def aggregate_wer_scores(scores: list[dict[str, Any]], model_ids: list[str], datapoint_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_scores = [score for score in scores if score["stt_model_id"] == model_id]
        if len(model_scores) != datapoint_count:
            continue
        substitutions = sum(score["wer"]["substitutions"] for score in model_scores)
        deletions = sum(score["wer"]["deletions"] for score in model_scores)
        insertions = sum(score["wer"]["insertions"] for score in model_scores)
        reference_words = sum(score["wer"]["reference_words"] for score in model_scores)
        hypothesis_words = sum(score["wer"]["hypothesis_words"] for score in model_scores)
        rows.append(
            {
                "Model": model_id,
                "WER": divide(substitutions + deletions + insertions, reference_words),
                "Reference Words": reference_words,
                "Hypothesis Words": hypothesis_words,
                "Substitutions": substitutions,
                "Deletions": deletions,
                "Insertions": insertions,
            }
        )
    return rows


def aggregate_wer_score_rows(scores: list[dict[str, Any]]) -> dict[str, Any]:
    substitutions = sum(score["wer"]["substitutions"] for score in scores)
    deletions = sum(score["wer"]["deletions"] for score in scores)
    insertions = sum(score["wer"]["insertions"] for score in scores)
    reference_words = sum(score["wer"]["reference_words"] for score in scores)
    hypothesis_words = sum(score["wer"]["hypothesis_words"] for score in scores)
    model_ids = sorted({str(score.get("stt_model_id", "")) for score in scores if score.get("stt_model_id")})
    return {
        "model_id": model_ids[0] if len(model_ids) == 1 else None,
        "datapoint_count": len(scores),
        "wer_reference": FORMAT_INVARIANT_WER_REFERENCE,
        "wer": divide(substitutions + deletions + insertions, reference_words),
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def aggregate_transcript_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transcript_lengths = [len(str(row.get("model_transcript", ""))) for row in rows]
    return {
        "datapoint_count": len(rows),
        "empty_transcript_count": sum(1 for length in transcript_lengths if length == 0),
        "total_transcript_characters": sum(transcript_lengths),
        "mean_transcript_characters": divide(sum(transcript_lengths), len(transcript_lengths)),
    }


def divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def entity_type_sort_key(entity_type: str) -> tuple[int, str]:
    try:
        return ENTITY_TYPES.index(entity_type), entity_type
    except ValueError:
        return 10_000, entity_type


def write_entity_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered_headers = [entity_level_column(entity_type) for entity_type in ENTITY_TYPES]
    entity_headers = [
        header
        for header in ordered_headers
        if any(header in row for row in rows)
    ]
    existing_rows, existing_fieldnames = read_csv(path)
    existing_entity_headers = [
        header
        for header in existing_fieldnames
        if header in ordered_headers
    ]
    fieldnames = ["Model", "TSR", "CTEM"]
    for header in [*existing_entity_headers, *entity_headers]:
        if header not in fieldnames:
            fieldnames.append(header)
    existing_wer = existing_metric_by_model(path, "WER")
    if existing_wer or "WER" in existing_fieldnames:
        fieldnames.append("WER")
    out_rows_by_model = {
        str(row["Model"]): {key: value for key, value in row.items() if key != "WER Reference"}
        for row in existing_rows
        if row.get("Model")
    }
    for row in rows:
        model_id = str(row["Model"])
        merged_row = dict(out_rows_by_model.get(model_id, {"Model": model_id}))
        merged_row.update(row)
        if "WER" in fieldnames:
            merged_row["WER"] = existing_wer.get(model_id, merged_row.get("WER", 0))
        out_rows_by_model[model_id] = merged_row
    ordered_models = [str(row["Model"]) for row in existing_rows if row.get("Model")]
    for row in rows:
        model_id = str(row["Model"])
        if model_id not in ordered_models:
            ordered_models.append(model_id)
    write_csv(path, [out_rows_by_model[model_id] for model_id in ordered_models], fieldnames)


def write_wer_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    existing_rows, existing_fieldnames = read_csv(path)
    if existing_rows and "Model" in existing_fieldnames:
        wer_by_model = {str(row["Model"]): row["WER"] for row in rows}
        fieldnames = [fieldname for fieldname in existing_fieldnames if fieldname != "WER Reference"]
        if "WER" not in fieldnames:
            fieldnames.append("WER")
        known_models = {str(row["Model"]) for row in existing_rows}
        out_rows = []
        for row in existing_rows:
            out_row = dict(row)
            out_row.pop("WER Reference", None)
            model_id = str(out_row["Model"])
            if model_id in wer_by_model:
                out_row["WER"] = wer_by_model[model_id]
            out_rows.append(out_row)
        for row in rows:
            model_id = str(row["Model"])
            if model_id not in known_models:
                out_rows.append(dict(row))
        write_csv(path, out_rows, fieldnames)
        return

    fieldnames = [
        "Model",
        "WER",
        "Reference Words",
        "Hypothesis Words",
        "Substitutions",
        "Deletions",
        "Insertions",
    ]
    write_csv(path, rows, fieldnames)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def existing_metric_by_model(path: Path, metric: str) -> dict[str, str]:
    rows, fieldnames = read_csv(path)
    if metric not in fieldnames:
        return {}
    return {str(row["Model"]): row[metric] for row in rows if row.get("Model")}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0) for field in fieldnames})
