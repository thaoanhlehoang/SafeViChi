"""Validated readers and deterministic stratified sampling for source data."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import heapq
import math
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Iterator, Mapping, Sequence

from .constants import (
    VIHSD_BINARY_LABELS,
    VIHSD_MAIN_REVISION,
    VIHSD_ORIGINAL_LABELS,
    VIHSD_REPO_ID,
    VOZ_MAIN_REVISION,
    VOZ_ORIGINAL_LABELS,
    VOZ_PARQUET_REVISION,
    VOZ_PARQUET_SHARDS,
    VOZ_REPO_ID,
)
from .schema import RejectedRecord, SourceRecord


class SourceDataError(RuntimeError):
    """Raised when a pinned source does not satisfy the build contract."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text(text: str) -> str:
    """Conservative exact-deduplication form; no lexical content is removed."""

    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) not in {"Cf", "Cc"}
    )
    return re.sub(r"\s+", " ", normalized)


@dataclass(frozen=True)
class ViHSDLoadResult:
    records: tuple[SourceRecord, ...]
    rejected: tuple[RejectedRecord, ...]
    raw_label_counts: Mapping[str, int]
    mapped_label_counts: Mapping[str, int]
    valid_label_counts: Mapping[str, int]


def load_vihsd_train(
    path: str | Path,
    *,
    conflict_policy: str = "quarantine",
) -> ViHSDLoadResult:
    """Load ViHSD train and quarantine semantically uncheckable label conflicts.

    ViHSD uses 0=CLEAN, 1=OFFENSIVE, 2=HATE.  Exact duplicate source
    strings that map to both binary labels cannot satisfy label preservation;
    the recommended policy therefore quarantines every row in such a group.
    """

    if conflict_policy not in {"quarantine", "keep"}:
        raise ValueError("conflict_policy must be 'quarantine' or 'keep'")
    source_path = Path(path)
    if not source_path.is_file():
        raise SourceDataError(f"ViHSD train file not found: {source_path}")

    parsed: list[SourceRecord] = []
    raw_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["free_text", "label_id"]:
            raise SourceDataError(
                "Unexpected ViHSD schema. Expected ['free_text', 'label_id'], "
                f"got {reader.fieldnames!r}."
            )
        for row_index, row in enumerate(reader):
            try:
                numeric_label = int(row["label_id"])
                original_label = VIHSD_ORIGINAL_LABELS[numeric_label]
                target_label = VIHSD_BINARY_LABELS[numeric_label]
            except (KeyError, TypeError, ValueError) as error:
                raise SourceDataError(
                    f"Invalid ViHSD label at zero-based row {row_index}: "
                    f"{row.get('label_id')!r}"
                ) from error
            text = row["free_text"] or ""
            raw_counts[original_label] += 1
            mapped_counts[target_label] += 1
            parsed.append(
                SourceRecord(
                    source_dataset=VIHSD_REPO_ID,
                    source_revision=VIHSD_MAIN_REVISION,
                    source_artifact_revision=VIHSD_MAIN_REVISION,
                    source_split="train",
                    source_row_id=str(row_index),
                    text=text,
                    label_original=original_label,
                    label=target_label,
                    annotation_type="human",
                    source_artifact="train.csv",
                )
            )

    canonical_labels: dict[str, set[str]] = defaultdict(set)
    for record in parsed:
        canonical_labels[canonical_text(record.text)].add(record.label)

    valid: list[SourceRecord] = []
    rejected: list[RejectedRecord] = []
    for record in parsed:
        key = canonical_text(record.text)
        reason: str | None = None
        if not key:
            reason = "empty_source_text"
        elif conflict_policy == "quarantine" and len(canonical_labels[key]) > 1:
            reason = "conflicting_binary_labels_for_exact_duplicate"
        if reason is None:
            valid.append(record)
        else:
            rejected.append(
                RejectedRecord(
                    source_dataset=record.source_dataset,
                    source_split=record.source_split,
                    source_row_id=record.source_row_id,
                    label_original=record.label_original,
                    label=record.label,
                    text_sha256=sha256_text(record.text),
                    reason=reason,
                )
            )

    return ViHSDLoadResult(
        records=tuple(valid),
        rejected=tuple(rejected),
        raw_label_counts=dict(raw_counts),
        mapped_label_counts=dict(mapped_counts),
        valid_label_counts=dict(Counter(record.label for record in valid)),
    )


def _open_parquet(location: str):
    import pyarrow.parquet as pq

    if location.startswith(("http://", "https://")):
        import fsspec

        # VOZ-HSD is public. The pinned URL prevents silent source drift.
        handle = fsspec.open(
            location,
            mode="rb",
            block_size=8 * 1024 * 1024,
            cache_type="readahead",
        ).open()
        return pq.ParquetFile(handle), handle
    handle = Path(location).open("rb")
    return pq.ParquetFile(handle), handle


def iter_voz_hsd(
    locations: Sequence[str] | None = None,
    *,
    batch_size: int = 16_384,
) -> Iterator[SourceRecord]:
    """Stream pinned VOZ-HSD Parquet columns without loading 10.7M rows."""

    locations = tuple(locations or VOZ_PARQUET_SHARDS)
    expected_columns = {"texts", "labels", "probs"}
    for shard_index, location in enumerate(locations):
        parquet_file, handle = _open_parquet(location)
        try:
            columns = set(parquet_file.schema_arrow.names)
            if columns != expected_columns:
                raise SourceDataError(
                    f"Unexpected VOZ-HSD schema in shard {shard_index}: {sorted(columns)!r}"
                )
            row_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=["texts", "labels", "probs"]
            ):
                texts = batch.column(0).to_pylist()
                labels = batch.column(1).to_pylist()
                probabilities = batch.column(2).to_pylist()
                for local_index, (text, numeric_label, probability) in enumerate(
                    zip(texts, labels, probabilities)
                ):
                    if numeric_label not in VOZ_ORIGINAL_LABELS:
                        raise SourceDataError(
                            f"Invalid VOZ-HSD label at shard {shard_index}, "
                            f"row {row_offset + local_index}: {numeric_label!r}"
                        )
                    label = VOZ_ORIGINAL_LABELS[numeric_label]
                    yield SourceRecord(
                        source_dataset=VOZ_REPO_ID,
                        source_revision=VOZ_MAIN_REVISION,
                        source_artifact_revision=VOZ_PARQUET_REVISION,
                        source_split="train",
                        source_row_id=f"{shard_index}:{row_offset + local_index}",
                        text=text or "",
                        label_original=label,
                        label=label,
                        annotation_type="weak_ai",
                        source_confidence=float(probability) if probability is not None else None,
                        source_artifact=Path(location).name,
                    )
                row_offset += len(texts)
        finally:
            handle.close()


def _priority(seed: int, record: SourceRecord) -> int:
    material = (
        f"{seed}\0{record.source_dataset}\0{record.source_artifact_revision}\0"
        f"{record.source_row_id}\0{sha256_text(record.text)}"
    )
    return int.from_bytes(
        hashlib.blake2b(material.encode("utf-8"), digest_size=16).digest(), "big"
    )


class _PriorityReservoir:
    """Keep the k smallest deterministic hashes in constant memory."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.seed = seed
        self.heap: list[tuple[int, str, SourceRecord]] = []

    def add(self, record: SourceRecord) -> None:
        rank = _priority(self.seed, record)
        item = (-rank, record.source_row_id, record)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif rank < -self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def ranked(self) -> list[tuple[int, SourceRecord]]:
        return sorted((-negative, record) for negative, _, record in self.heap)


@dataclass(frozen=True)
class VOZSampleResult:
    records: tuple[SourceRecord, ...]
    stats: Mapping[str, object]


def sample_voz_records(
    records: Iterable[SourceRecord],
    target_counts: Mapping[str, int],
    *,
    seed: int,
    min_confidence: float = 0.90,
    oversample_factor: float = 1.25,
    max_rows: int | None = None,
    exclude_canonical: Mapping[str, str] | None = None,
) -> VOZSampleResult:
    """One-pass, deterministic, stratified priority-reservoir sampling."""

    if set(target_counts) != {"HATE", "CLEAN"}:
        raise ValueError("target_counts must contain exactly HATE and CLEAN")
    if not 0.5 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0.5, 1.0]")
    if oversample_factor < 1.0:
        raise ValueError("oversample_factor must be >= 1.0")

    excluded = dict(exclude_canonical or {})
    reservoirs = {
        label: _PriorityReservoir(
            max(target, math.ceil(target * oversample_factor)), seed
        )
        for label, target in target_counts.items()
    }
    scanned = 0
    eligible: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    scan_complete = True
    for record in records:
        if max_rows is not None and scanned >= max_rows:
            scan_complete = False
            break
        scanned += 1
        if record.label not in reservoirs:
            rejected["unexpected_label"] += 1
            continue
        if not canonical_text(record.text):
            rejected["empty_source_text"] += 1
            continue
        if record.source_confidence is None or record.source_confidence < min_confidence:
            rejected["below_confidence_threshold"] += 1
            continue
        eligible[record.label] += 1
        reservoirs[record.label].add(record)

    selected: list[SourceRecord] = []
    selected_counts: Counter[str] = Counter()
    seen_labels = dict(excluded)
    skipped_duplicates: Counter[str] = Counter()
    for label in ("HATE", "CLEAN"):
        for _, record in reservoirs[label].ranked():
            key = canonical_text(record.text)
            existing_label = seen_labels.get(key)
            if existing_label is not None:
                reason = (
                    "exact_duplicate_label_conflict"
                    if existing_label != label
                    else "exact_duplicate"
                )
                skipped_duplicates[reason] += 1
                continue
            seen_labels[key] = label
            selected.append(record)
            selected_counts[label] += 1
            if selected_counts[label] >= target_counts[label]:
                break

    actual_counts = Counter(record.label for record in selected)
    shortages = {
        label: target_counts[label] - actual_counts[label]
        for label in target_counts
        if actual_counts[label] < target_counts[label]
    }
    if shortages:
        raise SourceDataError(
            "VOZ-HSD sampling could not satisfy unique class quotas: "
            f"{shortages}. Increase --voz-max-rows or --oversample-factor."
        )

    selected.sort(key=lambda record: (record.label, _priority(seed, record)))
    return VOZSampleResult(
        records=tuple(selected),
        stats={
            "rows_scanned": scanned,
            "scan_complete": scan_complete,
            "min_confidence": min_confidence,
            "eligible_by_label": dict(eligible),
            "selected_by_label": dict(actual_counts),
            "rejected": dict(rejected),
            "skipped_selected_duplicates": dict(skipped_duplicates),
            "sampling_method": "stratified_deterministic_priority_reservoir",
            "oversample_factor": oversample_factor,
        },
    )


__all__ = [
    "SourceDataError",
    "VOZSampleResult",
    "ViHSDLoadResult",
    "canonical_text",
    "iter_voz_hsd",
    "load_vihsd_train",
    "sample_voz_records",
    "sha256_text",
]
