"""Automated gates and a deterministic human-review worksheet."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Mapping, Sequence

from src.variant_generator.controlled import (
    BASE_PERTURBATION_TYPES,
    Edit,
    replay_edits,
)

from .sources import canonical_text


class QualityGateError(RuntimeError):
    """Raised when a built corpus violates a hard invariant."""


def _nested_counts(rows: Sequence[Mapping[str, object]], *keys: str) -> dict[str, object]:
    if len(keys) == 1:
        return dict(Counter(str(row[keys[0]]) for row in rows))
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[keys[0]])].append(row)
    return {
        key: _nested_counts(group_rows, *keys[1:])
        for key, group_rows in sorted(grouped.items())
    }


def _similarity(left: str, right: str) -> float:
    def fold(text: str) -> str:
        value = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
        value = "".join(char for char in value if not unicodedata.combining(char))
        return "".join(char for char in value if char.isalnum())

    return SequenceMatcher(None, fold(left), fold(right)).ratio()


def run_automated_qa(
    rows: Sequence[Mapping[str, object]],
    *,
    target_label_counts: Mapping[str, int],
    require_full_type_coverage: bool = True,
) -> dict[str, object]:
    """Validate hard invariants and return aggregate audit evidence."""

    errors: list[str] = []
    labels = Counter(str(row.get("label")) for row in rows)
    expected_total = sum(target_label_counts.values())
    if len(rows) != expected_total:
        errors.append(f"row_count={len(rows)} but expected {expected_total}")
    if dict(labels) != dict(target_label_counts):
        errors.append(
            f"label_counts={dict(labels)!r} but expected {dict(target_label_counts)!r}"
        )

    group_splits: dict[str, str] = {}
    output_splits: dict[str, str] = {}
    output_counts: Counter[str] = Counter()
    output_duplicates = 0
    weak_eval_rows = 0
    changed_rows = 0
    replayable_rows = 0
    similarities: list[float] = []
    type_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"row-{index}"))
        original = str(row.get("original_text", ""))
        perturbed = str(row.get("text", ""))
        if original != perturbed:
            changed_rows += 1
        else:
            errors.append(f"{sample_id}: text was not changed")
        if not perturbed.strip():
            errors.append(f"{sample_id}: perturbed text is empty")

        edit_dicts = row.get("perturbation_edits")
        if not isinstance(edit_dicts, list) or not edit_dicts:
            errors.append(f"{sample_id}: missing perturbation edit trace")
        else:
            try:
                edits = [Edit(**edit) for edit in edit_dicts]
                if replay_edits(original, edits) != perturbed:
                    raise ValueError("replayed output differs")
                replayable_rows += 1
            except (TypeError, ValueError) as error:
                errors.append(f"{sample_id}: invalid edit trace ({error})")

        types = row.get("perturbation_types")
        if not isinstance(types, list) or not types:
            errors.append(f"{sample_id}: perturbation_types is empty")
        else:
            type_counts.update(str(item) for item in types)
        mode = str(row.get("perturbation_mode"))
        mode_counts[mode] += 1
        if mode == "mixed" and isinstance(types, list) and len(set(types)) < 2:
            errors.append(f"{sample_id}: mixed mode contains fewer than two types")

        if int(row.get("perturbation_count", -1)) != len(edit_dicts or []):
            errors.append(f"{sample_id}: perturbation_count does not match edit trace")
        if row.get("label") != row.get("label_mapped"):
            errors.append(f"{sample_id}: label and label_mapped differ")
        if row.get("label") not in {"HATE", "CLEAN"}:
            errors.append(f"{sample_id}: unexpected binary label {row.get('label')!r}")

        split = str(row.get("target_split"))
        group_id = str(row.get("duplicate_group_id"))
        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            errors.append(f"{sample_id}: duplicate group crosses target splits")
        if row.get("annotation_type") == "weak_ai" and split != "train":
            weak_eval_rows += 1

        output_key = canonical_text(perturbed)
        previous_output_split = output_splits.setdefault(output_key, split)
        if previous_output_split != split:
            errors.append(f"{sample_id}: exact perturbed output leaks across splits")
        output_duplicates += int(output_counts[output_key] > 0)
        output_counts[output_key] += 1

        similarities.append(_similarity(original, perturbed))

    missing_types = sorted(set(BASE_PERTURBATION_TYPES) - set(type_counts))
    if require_full_type_coverage and missing_types:
        errors.append(f"missing perturbation types: {missing_types}")
    if require_full_type_coverage and not mode_counts.get("mixed"):
        errors.append("no mixed perturbation samples were generated")
    if weak_eval_rows:
        errors.append(f"{weak_eval_rows} weak-label rows are present in validation/test")

    similarities_sorted = sorted(similarities)

    def percentile(fraction: float) -> float | None:
        if not similarities_sorted:
            return None
        index = round((len(similarities_sorted) - 1) * fraction)
        return round(similarities_sorted[index], 6)

    report: dict[str, object] = {
        "status": "passed" if not errors else "failed",
        "errors": errors[:100],
        "error_count": len(errors),
        "row_count": len(rows),
        "changed_row_count": changed_rows,
        "replayable_trace_count": replayable_rows,
        "label_counts": dict(labels),
        "source_label_counts": _nested_counts(rows, "source_dataset", "label"),
        "split_label_counts": _nested_counts(rows, "target_split", "label"),
        "annotation_split_counts": _nested_counts(rows, "annotation_type", "target_split"),
        "perturbation_type_counts": dict(type_counts),
        "perturbation_mode_counts": dict(mode_counts),
        "missing_perturbation_types": missing_types,
        "weak_rows_in_evaluation": weak_eval_rows,
        "exact_perturbed_duplicate_rows": output_duplicates,
        "surface_similarity": {
            "min": round(min(similarities), 6) if similarities else None,
            "p05": percentile(0.05),
            "median": percentile(0.50),
            "p95": percentile(0.95),
            "max": round(max(similarities), 6) if similarities else None,
        },
        "automatic_scope_note": (
            "Automatic checks cover structural invariants, trace replay, label-source "
            "separation, leakage, and surface similarity. Semantic/label preservation "
            "and naturalness still require the generated human-review worksheet."
        ),
    }
    if errors:
        raise QualityGateError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def select_human_review_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    size: int,
    seed: int,
) -> list[Mapping[str, object]]:
    """Select a reproducible worksheet sample with categorical coverage."""

    if size <= 0:
        return []

    def rank(row: Mapping[str, object]) -> str:
        material = f"{seed}\0{row['sample_id']}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    ordered = sorted(rows, key=rank)
    selected: list[Mapping[str, object]] = []
    selected_ids: set[str] = set()

    def add_first(predicate) -> None:
        for row in ordered:
            sample_id = str(row["sample_id"])
            if sample_id not in selected_ids and predicate(row):
                selected.append(row)
                selected_ids.add(sample_id)
                return

    for label in ("HATE", "CLEAN"):
        add_first(lambda row, value=label: row["label"] == value)
    for source in sorted({str(row["source_dataset"]) for row in rows}):
        add_first(lambda row, value=source: row["source_dataset"] == value)
    for split in ("train", "validation", "test"):
        add_first(lambda row, value=split: row["target_split"] == value)
    for perturbation_type in BASE_PERTURBATION_TYPES:
        add_first(
            lambda row, value=perturbation_type: value in row["perturbation_types"]
        )
    add_first(lambda row: row["perturbation_mode"] == "mixed")

    for row in ordered:
        if len(selected) >= min(size, len(rows)):
            break
        sample_id = str(row["sample_id"])
        if sample_id not in selected_ids:
            selected.append(row)
            selected_ids.add(sample_id)
    return selected[:size]


def write_human_review_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    fields = [
        "sample_id",
        "source_dataset",
        "target_split",
        "annotation_type",
        "label",
        "original_text",
        "perturbed_text",
        "perturbation_types",
        "random_seed",
        "naturalness_1_to_5",
        "semantic_preservation_yes_no",
        "label_preservation_yes_no",
        "reviewer_notes",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_dataset": row["source_dataset"],
                    "target_split": row["target_split"],
                    "annotation_type": row["annotation_type"],
                    "label": row["label"],
                    "original_text": row["original_text"],
                    "perturbed_text": row["text"],
                    "perturbation_types": json.dumps(
                        row["perturbation_types"], ensure_ascii=False
                    ),
                    "random_seed": row["random_seed"],
                    "naturalness_1_to_5": "",
                    "semantic_preservation_yes_no": "",
                    "label_preservation_yes_no": "",
                    "reviewer_notes": "",
                }
            )


__all__ = [
    "QualityGateError",
    "run_automated_qa",
    "select_human_review_rows",
    "write_human_review_csv",
]
