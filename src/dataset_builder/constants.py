"""Pinned source identifiers and target label policy."""

from __future__ import annotations

VIHSD_REPO_ID = "uitnlp/vihsd"
VIHSD_MAIN_REVISION = "88e81b36ca376867640dad9df295e0f7d499ab80"
VIHSD_PARQUET_REVISION = "a63883c6a64e473b741966a6ae1b1de693334a56"

VOZ_REPO_ID = "tarudesu/VOZ-HSD"
VOZ_MAIN_REVISION = "923915f5f633c503babf2c6dcf7003593318b04f"
VOZ_PARQUET_REVISION = "2518cc1fd5287975fbc6f842fb29625880c012f4"
VOZ_PARQUET_SHARDS = tuple(
    "https://huggingface.co/datasets/"
    f"{VOZ_REPO_ID}/resolve/{VOZ_PARQUET_REVISION}/default/train/{index:04d}.parquet"
    for index in range(4)
)

VIHSD_ORIGINAL_LABELS = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
VIHSD_BINARY_LABELS = {0: "CLEAN", 1: "HATE", 2: "HATE"}
VOZ_ORIGINAL_LABELS = {0: "CLEAN", 1: "HATE"}
LABEL_IDS = {"CLEAN": 0, "HATE": 1}

# Requested final composition: ViHSD HATE+OFFENSIVE plus 31K weak HATE,
# and ViHSD CLEAN plus 20K weak CLEAN.
TARGET_LABEL_COUNTS = {"HATE": 35_162, "CLEAN": 39_886}
TARGET_TOTAL = sum(TARGET_LABEL_COUNTS.values())

DATASET_VERSION = "1.2.0"
