from __future__ import annotations

import json
import math

from src.dataset_builder.build import augment_records
from src.dataset_builder.constants import VIHSD_MAIN_REVISION, VIHSD_REPO_ID
from src.dataset_builder.quality import run_automated_qa
from src.dataset_builder.schema import SourceRecord
from src.variant_generator.controlled import (
    ControlledPerturber,
    DEFAULT_TEENCODE_SPAN_RATE,
    load_teencode_lexicon,
    replay_edits,
)
from src.variant_generator.teencode import compile_teencode_lexicon


def test_default_dictionary_is_compiled_with_conflicts_quarantined() -> None:
    lexicon = load_teencode_lexicon()

    assert lexicon.input_pair_count == 643
    assert lexicon.accepted_pair_count == 611
    assert lexicon.standard_form_count == 283
    assert lexicon.excluded_reason_counts == {
        "curated_conflict": 2,
        "curated_duplicate": 20,
        "normalization_chain": 10,
    }
    assert lexicon.normalizes_pair("0", "không")
    assert not lexicon.normalizes_pair("dai", "trai")
    assert not lexicon.normalizes_pair("nc", "nước")


def test_compilation_is_independent_of_json_entry_order(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    entries = {"hok": "không", "bik": "biết", "bit": "biết"}
    first_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    second_path.write_text(
        json.dumps(dict(reversed(tuple(entries.items()))), ensure_ascii=False),
        encoding="utf-8",
    )

    first = compile_teencode_lexicon(first_path)
    second = compile_teencode_lexicon(second_path)

    assert first.variants_by_standard == second.variants_by_standard
    assert first.standard_by_variant == second.standard_by_variant


def test_teencode_edit_has_replayable_resource_provenance() -> None:
    perturber = ControlledPerturber()
    result = perturber.perturb(
        "Tôi không biết gì.",
        "CLEAN",
        seed=20260903,
        required_type="teencode_lexical",
        min_operations=1,
        max_operations=1,
    )

    assert set(result.perturbation_types) == {"teencode_lexical"}
    assert result.teencode_valid_span_count >= 2
    assert result.teencode_applied_span_count == math.ceil(
        result.teencode_valid_span_count * result.teencode_span_target_rate
    )
    assert replay_edits(result.original_text, result.edits) == result.perturbed_text
    edit = result.edits[0]
    assert edit.resource_name == "teencode_dict"
    assert edit.resource_version == perturber.teencode_lexicon.version
    assert edit.resource_sha256 == perturber.teencode_lexicon.source_sha256
    assert perturber.teencode_lexicon.normalizes_pair(edit.after, edit.before)
    assert edit.rule_id == perturber.teencode_lexicon.rule_id(edit.before, edit.after)


def test_teencode_matching_respects_protected_social_spans() -> None:
    perturber = ControlledPerturber()

    assert not perturber.is_teencode_eligible("#không", "CLEAN")
    assert not perturber.is_teencode_eligible("https://example.test/không", "CLEAN")


def test_augmentation_targets_every_eligible_sample_and_all_valid_spans() -> None:
    records = tuple(
        SourceRecord(
            source_dataset=VIHSD_REPO_ID,
            source_revision=VIHSD_MAIN_REVISION,
            source_artifact_revision=VIHSD_MAIN_REVISION,
            source_split="train",
            source_row_id=str(index),
            text=f"Tôi không biết gì về nội dung kiểm thử thứ {index}.",
            label_original="CLEAN",
            label="CLEAN",
            annotation_type="human",
            source_artifact="train.csv",
        )
        for index in range(20)
    )
    groups = tuple(f"group-{index}" for index in range(len(records)))
    splits = ("train",) * len(records)

    rows = augment_records(
        records,
        groups,
        splits,
        master_seed=20260903,
    )

    assert sum(bool(row["teencode_eligible"]) for row in rows) == 20
    assert sum(bool(row["teencode_targeted"]) for row in rows) == 20
    assert sum(bool(row["teencode_applied"]) for row in rows) == 20
    assert all(
        bool(row["teencode_targeted"])
        == ("teencode_lexical" in row["perturbation_types"])
        for row in rows
    )
    assert all(
        int(row["teencode_applied_span_count"])
        == int(row["teencode_valid_span_count"])
        for row in rows
    )
    report = run_automated_qa(
        rows,
        target_label_counts={"CLEAN": len(rows)},
        teencode_target_rate=1.0,
        teencode_span_target_rate=DEFAULT_TEENCODE_SPAN_RATE,
        require_full_type_coverage=False,
    )
    assert report["teencode_lexical"]["eligible_count"] == 20
    assert report["teencode_lexical"]["applied_count"] == 20
    assert report["teencode_lexical"]["span_application_rate"] == 1.0
