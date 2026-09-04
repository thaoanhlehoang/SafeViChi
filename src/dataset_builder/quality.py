"""Automated gates and a deterministic human-review worksheet."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Mapping, Sequence

from src.variant_generator.controlled import (
    BASE_PERTURBATION_TYPES,
    Edit,
    load_teencode_lexicon,
    replay_edits,
)
from src.variant_generator.teencode import TeencodeLexicon

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
    teencode_lexicon: TeencodeLexicon | None = None,
    teencode_target_rate: float | None = None,
    teencode_span_target_rate: float | None = None,
) -> dict[str, object]:
    """Validate hard invariants and return aggregate audit evidence."""

    errors: list[str] = []
    lexicon = teencode_lexicon or load_teencode_lexicon()
    if teencode_target_rate is not None and not 0.0 <= teencode_target_rate <= 1.0:
        raise ValueError("teencode_target_rate must be between 0 and 1")
    if teencode_span_target_rate is not None and not (
        0.0 < teencode_span_target_rate <= 1.0
    ):
        raise ValueError(
            "teencode_span_target_rate must be greater than 0 and at most 1"
        )
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
    teencode_eligible_by_stratum: Counter[tuple[str, str, str]] = Counter()
    teencode_targeted_by_stratum: Counter[tuple[str, str, str]] = Counter()
    teencode_applied_by_stratum: Counter[tuple[str, str, str]] = Counter()
    teencode_valid_spans_by_stratum: Counter[tuple[str, str, str]] = Counter()
    teencode_applied_spans_by_stratum: Counter[tuple[str, str, str]] = Counter()
    teencode_canonical_forms: Counter[str] = Counter()
    teencode_variants: Counter[str] = Counter()
    teencode_edit_count = 0

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

        row_teencode_edit_count = 0
        edit_dicts = row.get("perturbation_edits")
        if not isinstance(edit_dicts, list) or not edit_dicts:
            errors.append(f"{sample_id}: missing perturbation edit trace")
        else:
            try:
                edits = [Edit(**edit) for edit in edit_dicts]
                if replay_edits(original, edits) != perturbed:
                    raise ValueError("replayed output differs")
                for edit in edits:
                    if edit.perturbation_type != "teencode_lexical":
                        continue
                    teencode_edit_count += 1
                    row_teencode_edit_count += 1
                    if edit.resource_name != "teencode_dict":
                        errors.append(
                            f"{sample_id}: teencode edit has invalid resource_name"
                        )
                    if edit.resource_version != lexicon.version:
                        errors.append(
                            f"{sample_id}: teencode edit has a resource version mismatch"
                        )
                    if edit.resource_sha256 != lexicon.source_sha256:
                        errors.append(
                            f"{sample_id}: teencode edit has a resource hash mismatch"
                        )
                    canonical = edit.canonical_form or edit.before
                    variant = edit.selected_variant or edit.after
                    if not lexicon.normalizes_pair(variant, canonical):
                        errors.append(
                            f"{sample_id}: teencode edit does not round-trip through the lexicon"
                        )
                    if edit.rule_id != lexicon.rule_id(canonical, variant):
                        errors.append(
                            f"{sample_id}: teencode edit rule_id is inconsistent"
                        )
                    if not edit.candidate_count or edit.candidate_count < 1:
                        errors.append(
                            f"{sample_id}: teencode edit is missing candidate_count"
                        )
                    teencode_canonical_forms[canonical.casefold()] += 1
                    teencode_variants[variant.casefold()] += 1
                replayable_rows += 1
            except (TypeError, ValueError) as error:
                errors.append(f"{sample_id}: invalid edit trace ({error})")

        types = row.get("perturbation_types")
        type_values = [str(item) for item in types] if isinstance(types, list) else []
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
        stratum = (str(row.get("source_dataset")), str(row.get("label")), split)
        teencode_eligible = bool(row.get("teencode_eligible", False))
        teencode_targeted = bool(row.get("teencode_targeted", False))
        teencode_applied = bool(row.get("teencode_applied", False))
        teencode_valid_span_count = int(row.get("teencode_valid_span_count", 0))
        teencode_applied_span_count = int(
            row.get("teencode_applied_span_count", 0)
        )
        if teencode_eligible != (teencode_valid_span_count > 0):
            errors.append(
                f"{sample_id}: teencode eligibility disagrees with valid span count"
            )
        if teencode_applied != (teencode_applied_span_count > 0):
            errors.append(
                f"{sample_id}: teencode application disagrees with span count"
            )
        if teencode_applied_span_count != row_teencode_edit_count:
            errors.append(
                f"{sample_id}: teencode applied span count disagrees with edit trace"
            )
        if teencode_targeted:
            expected_span_count = (
                max(
                    1,
                    math.ceil(
                        teencode_valid_span_count * teencode_span_target_rate
                    ),
                )
                if teencode_span_target_rate is not None
                else None
            )
            if (
                expected_span_count is not None
                and teencode_applied_span_count != expected_span_count
            ):
                errors.append(
                    f"{sample_id}: applied {teencode_applied_span_count} teencode "
                    f"spans but expected {expected_span_count}"
                )
        elif teencode_applied_span_count:
            errors.append(f"{sample_id}: non-targeted row has teencode span edits")
        if teencode_eligible:
            teencode_eligible_by_stratum[stratum] += 1
        if teencode_targeted:
            teencode_targeted_by_stratum[stratum] += 1
        if teencode_applied:
            teencode_applied_by_stratum[stratum] += 1
        if teencode_targeted:
            teencode_valid_spans_by_stratum[stratum] += teencode_valid_span_count
        teencode_applied_spans_by_stratum[stratum] += teencode_applied_span_count
        type_reports_applied = "teencode_lexical" in type_values
        if teencode_applied != type_reports_applied:
            errors.append(f"{sample_id}: teencode_applied disagrees with edit types")
        if teencode_targeted and not teencode_eligible:
            errors.append(f"{sample_id}: ineligible row was targeted for teencode")
        if teencode_applied != teencode_targeted:
            errors.append(f"{sample_id}: teencode target was not fulfilled exactly")

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

    expected_types = set(BASE_PERTURBATION_TYPES)
    if teencode_target_rate == 0.0:
        expected_types.discard("teencode_lexical")
    missing_types = sorted(expected_types - set(type_counts))
    saturated_teencode = (
        teencode_target_rate == 1.0 and teencode_span_target_rate == 1.0
    )
    coverage_exempt_types = missing_types if saturated_teencode else []
    blocking_missing_types = sorted(set(missing_types) - set(coverage_exempt_types))
    if require_full_type_coverage and blocking_missing_types:
        errors.append(f"missing perturbation types: {blocking_missing_types}")
    if require_full_type_coverage and not mode_counts.get("mixed"):
        errors.append("no mixed perturbation samples were generated")
    if weak_eval_rows:
        errors.append(f"{weak_eval_rows} weak-label rows are present in validation/test")

    all_teencode_strata = sorted(
        set(teencode_eligible_by_stratum)
        | set(teencode_targeted_by_stratum)
        | set(teencode_applied_by_stratum)
        | set(teencode_valid_spans_by_stratum)
        | set(teencode_applied_spans_by_stratum)
    )
    teencode_strata: list[dict[str, object]] = []
    for source, label, split in all_teencode_strata:
        key = (source, label, split)
        eligible_count = teencode_eligible_by_stratum[key]
        targeted_count = teencode_targeted_by_stratum[key]
        applied_count = teencode_applied_by_stratum[key]
        valid_span_count = teencode_valid_spans_by_stratum[key]
        applied_span_count = teencode_applied_spans_by_stratum[key]
        expected_target = (
            int(eligible_count * teencode_target_rate + 0.5)
            if teencode_target_rate is not None
            else None
        )
        if expected_target is not None and targeted_count != expected_target:
            errors.append(
                "teencode quota mismatch for "
                f"{source}/{label}/{split}: targeted={targeted_count}, "
                f"expected={expected_target}"
            )
        if applied_count != targeted_count:
            errors.append(
                "teencode application mismatch for "
                f"{source}/{label}/{split}: applied={applied_count}, "
                f"targeted={targeted_count}"
            )
        teencode_strata.append(
            {
                "source_dataset": source,
                "label": label,
                "target_split": split,
                "eligible_count": eligible_count,
                "targeted_count": targeted_count,
                "applied_count": applied_count,
                "expected_target_count": expected_target,
                "valid_span_count_in_targeted_rows": valid_span_count,
                "applied_span_count": applied_span_count,
                "span_application_rate": round(
                    applied_span_count / valid_span_count, 6
                )
                if valid_span_count
                else 0.0,
            }
        )

    teencode_eligible_count = sum(teencode_eligible_by_stratum.values())
    teencode_targeted_count = sum(teencode_targeted_by_stratum.values())
    teencode_applied_count = sum(teencode_applied_by_stratum.values())
    teencode_valid_span_count = sum(teencode_valid_spans_by_stratum.values())
    teencode_applied_span_count = sum(teencode_applied_spans_by_stratum.values())

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
        "coverage_exempt_types": coverage_exempt_types,
        "coverage_exemption_reason": (
            "Required teencode at 100% sample and span coverage consumes canonical "
            "spans before overlapping lexical transforms can use them; generator-level "
            "tests retain coverage of every supported transform."
            if coverage_exempt_types
            else None
        ),
        "teencode_lexical": {
            "requested_target_rate": teencode_target_rate,
            "requested_span_target_rate": teencode_span_target_rate,
            "eligible_count": teencode_eligible_count,
            "ineligible_count": len(rows) - teencode_eligible_count,
            "targeted_count": teencode_targeted_count,
            "applied_count": teencode_applied_count,
            "applied_among_eligible_rate": round(
                teencode_applied_count / teencode_eligible_count, 6
            )
            if teencode_eligible_count
            else 0.0,
            "applied_among_all_rows_rate": round(
                teencode_applied_count / len(rows), 6
            )
            if rows
            else 0.0,
            "edit_count": teencode_edit_count,
            "valid_span_count_in_targeted_rows": teencode_valid_span_count,
            "applied_span_count": teencode_applied_span_count,
            "span_application_rate": round(
                teencode_applied_span_count / teencode_valid_span_count, 6
            )
            if teencode_valid_span_count
            else 0.0,
            "distinct_canonical_forms": len(teencode_canonical_forms),
            "distinct_variants": len(teencode_variants),
            "by_stratum": teencode_strata,
        },
        "missing_perturbation_types": blocking_missing_types,
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
    add_first(
        lambda row: any(
            edit.get("perturbation_type") == "teencode_lexical"
            and len(str(edit.get("selected_variant", edit.get("after", "")))) <= 2
            for edit in row.get("perturbation_edits", [])
        )
    )
    add_first(
        lambda row: any(
            edit.get("perturbation_type") == "teencode_lexical"
            and int(edit.get("candidate_count", 0)) >= 10
            for edit in row.get("perturbation_edits", [])
        )
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
