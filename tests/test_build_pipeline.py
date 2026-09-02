from __future__ import annotations

import re

from src.dataset_builder.build import augment_records
from src.dataset_builder.constants import VIHSD_MAIN_REVISION, VIHSD_REPO_ID
from src.dataset_builder.quality import run_automated_qa
from src.dataset_builder.schema import SourceRecord


def test_augmentation_rows_use_portable_hex_seeds_and_pass_structural_qa() -> None:
    records = tuple(
        SourceRecord(
            source_dataset=VIHSD_REPO_ID,
            source_revision=VIHSD_MAIN_REVISION,
            source_artifact_revision=VIHSD_MAIN_REVISION,
            source_split="train",
            source_row_id=str(index),
            text=(
                "Bây giờ mình không biết nói chuyện với mọi người, "
                f"đây là nội dung kiểm thử duy nhất thứ {index}."
            ),
            label_original="CLEAN",
            label="CLEAN",
            annotation_type="human",
            source_artifact="train.csv",
        )
        for index in range(20)
    )
    groups = tuple(f"group-{index}" for index in range(len(records)))
    splits = ("train",) * len(records)

    rows = augment_records(records, groups, splits, master_seed=20260902)
    report = run_automated_qa(
        rows,
        target_label_counts={"CLEAN": len(rows)},
        require_full_type_coverage=False,
    )

    assert report["status"] == "passed"
    assert all(re.fullmatch(r"[0-9a-f]{16}", str(row["random_seed"])) for row in rows)
