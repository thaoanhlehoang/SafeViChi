from __future__ import annotations

from collections import Counter
import csv

from src.dataset_builder.constants import VOZ_MAIN_REVISION, VOZ_PARQUET_REVISION, VOZ_REPO_ID
from src.dataset_builder.schema import SourceRecord
from src.dataset_builder.sources import load_vihsd_train, sample_voz_records


def _voz(row_id: int, label: str, confidence: float = 0.99) -> SourceRecord:
    return SourceRecord(
        source_dataset=VOZ_REPO_ID,
        source_revision=VOZ_MAIN_REVISION,
        source_artifact_revision=VOZ_PARQUET_REVISION,
        source_split="train",
        source_row_id=f"0:{row_id}",
        text=f"Nội dung diễn đàn duy nhất số {row_id} cho lớp {label}",
        label_original=label,
        label=label,
        annotation_type="weak_ai",
        source_confidence=confidence,
        source_artifact="0000.parquet",
    )


def test_vihsd_mapping_and_conflict_quarantine(tmp_path) -> None:
    path = tmp_path / "train.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["free_text", "label_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"free_text": "cùng một câu", "label_id": 0},
                {"free_text": "CÙNG   MỘT CÂU", "label_id": 2},
                {"free_text": "", "label_id": 0},
                {"free_text": "một câu xúc phạm", "label_id": 1},
                {"free_text": "một câu sạch", "label_id": 0},
            ]
        )

    result = load_vihsd_train(path)
    assert result.raw_label_counts == {"CLEAN": 3, "HATE": 1, "OFFENSIVE": 1}
    assert result.mapped_label_counts == {"CLEAN": 3, "HATE": 2}
    assert result.valid_label_counts == {"HATE": 1, "CLEAN": 1}
    assert Counter(item.reason for item in result.rejected) == {
        "conflicting_binary_labels_for_exact_duplicate": 2,
        "empty_source_text": 1,
    }


def test_voz_priority_sampling_is_stratified_unique_and_reproducible() -> None:
    source = [_voz(i, "HATE" if i % 2 else "CLEAN") for i in range(100)]
    source.append(_voz(1000, "HATE", confidence=0.6))

    first = sample_voz_records(
        source,
        {"HATE": 8, "CLEAN": 7},
        seed=42,
        min_confidence=0.9,
        oversample_factor=2.0,
    )
    second = sample_voz_records(
        source,
        {"HATE": 8, "CLEAN": 7},
        seed=42,
        min_confidence=0.9,
        oversample_factor=2.0,
    )

    assert Counter(record.label for record in first.records) == {"HATE": 8, "CLEAN": 7}
    assert [record.source_row_id for record in first.records] == [
        record.source_row_id for record in second.records
    ]
    assert len({record.text for record in first.records}) == 15
    assert first.stats["rejected"] == {"below_confidence_threshold": 1}
