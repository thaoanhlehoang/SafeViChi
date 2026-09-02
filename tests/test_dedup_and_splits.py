from __future__ import annotations

from src.dataset_builder.constants import VIHSD_MAIN_REVISION, VIHSD_REPO_ID, VOZ_MAIN_REVISION, VOZ_PARQUET_REVISION, VOZ_REPO_ID
from src.dataset_builder.dedup import assign_leakage_safe_splits, group_near_duplicates
from src.dataset_builder.schema import SourceRecord


def _record(source: str, row_id: str, text: str, annotation: str) -> SourceRecord:
    return SourceRecord(
        source_dataset=source,
        source_revision=VIHSD_MAIN_REVISION if annotation == "human" else VOZ_MAIN_REVISION,
        source_artifact_revision=(
            VIHSD_MAIN_REVISION if annotation == "human" else VOZ_PARQUET_REVISION
        ),
        source_split="train",
        source_row_id=row_id,
        text=text,
        label_original="CLEAN",
        label="CLEAN",
        annotation_type=annotation,
    )


def test_near_duplicate_group_is_atomic_and_weak_group_is_train_only() -> None:
    records = [
        _record(VIHSD_REPO_ID, "0", "Xin chào người bạn của tôi", "human"),
        _record(VOZ_REPO_ID, "0:9", "xin chao nguoi ban cua toi", "weak_ai"),
        _record(VIHSD_REPO_ID, "1", "Một câu hoàn toàn khác biệt", "human"),
    ]
    groups = group_near_duplicates(records, jaccard_threshold=0.85)
    splits = assign_leakage_safe_splits(
        records,
        groups.group_ids,
        seed=12,
        human_validation_ratio=0.2,
        human_test_ratio=0.2,
    )

    assert groups.group_ids[0] == groups.group_ids[1]
    assert splits[0] == splits[1] == "train"
