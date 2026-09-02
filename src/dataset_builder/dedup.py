"""Exact/near-duplicate grouping and leakage-safe split assignment."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Mapping, Sequence

from .schema import SourceRecord
from .sources import canonical_text


def _dedup_form(text: str) -> str:
    value = canonical_text(text).replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", value)
    value = "".join(char for char in decomposed if not unicodedata.combining(char))
    value = "".join(char if char.isalnum() else " " for char in value)
    return re.sub(r"\s+", " ", value).strip()


def _shingles(text: str, n: int = 3) -> set[str]:
    compact = f"  {text}  "
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _simhash(features: set[str]) -> int:
    vector = [0] * 64
    for feature in features:
        hashed = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if hashed & (1 << bit) else -1
    result = 0
    for bit, value in enumerate(vector):
        if value >= 0:
            result |= 1 << bit
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


@dataclass(frozen=True)
class DuplicateGroups:
    group_ids: tuple[str, ...]
    stats: Mapping[str, object]


def group_near_duplicates(
    records: Sequence[SourceRecord],
    *,
    jaccard_threshold: float = 0.90,
    max_hamming_distance: int = 8,
    min_near_length: int = 12,
) -> DuplicateGroups:
    """Group exact and conservative near duplicates with SimHash LSH.

    LSH only proposes candidates. Character-trigram Jaccard and a length
    ratio then verify every near-duplicate edge to limit false merges.
    """

    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    union_find = _UnionFind(len(records))
    exact: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        exact[canonical_text(record.text)].append(index)

    exact_edges = 0
    for indices in exact.values():
        for index in indices[1:]:
            exact_edges += int(union_find.union(indices[0], index))

    symbolic: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = canonical_text(record.text).replace(" ", "")
        if key and not any(char.isalnum() for char in key):
            skeleton = re.sub(r"(.)\1+", r"\1", key)
            symbolic[skeleton].append(index)
    symbolic_edges = 0
    for indices in symbolic.values():
        for index in indices[1:]:
            symbolic_edges += int(union_find.union(indices[0], index))

    representatives = [indices[0] for indices in exact.values()]
    forms: dict[int, str] = {}
    feature_sets: dict[int, set[str]] = {}
    signatures: dict[int, int] = {}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    near_edges = 0

    for index in representatives:
        form = _dedup_form(records[index].text)
        if len(form) < min_near_length:
            continue
        features = _shingles(form)
        signature = _simhash(features)
        forms[index] = form
        feature_sets[index] = features
        signatures[index] = signature
        candidates: set[int] = set()
        for band in range(4):
            bucket_key = (band, (signature >> (band * 16)) & 0xFFFF)
            candidates.update(buckets[bucket_key])

        for other in candidates:
            length_ratio = min(len(form), len(forms[other])) / max(
                len(form), len(forms[other])
            )
            if length_ratio < 0.85:
                continue
            if (signature ^ signatures[other]).bit_count() > max_hamming_distance:
                continue
            if _jaccard(features, feature_sets[other]) < jaccard_threshold:
                continue
            near_edges += int(union_find.union(index, other))

        for band in range(4):
            bucket_key = (band, (signature >> (band * 16)) & 0xFFFF)
            buckets[bucket_key].append(index)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members[union_find.find(index)].append(index)

    root_to_id: dict[int, str] = {}
    for root, indices in members.items():
        fingerprints = sorted(
            hashlib.sha256(canonical_text(records[index].text).encode("utf-8")).hexdigest()
            for index in indices
        )
        digest = hashlib.sha256("\0".join(fingerprints).encode("ascii")).hexdigest()[:20]
        root_to_id[root] = f"grp_{digest}"

    group_ids = tuple(root_to_id[union_find.find(index)] for index in range(len(records)))
    sizes = Counter(group_ids)
    return DuplicateGroups(
        group_ids=group_ids,
        stats={
            "record_count": len(records),
            "group_count": len(sizes),
            "exact_union_edges": exact_edges,
            "symbolic_repetition_union_edges": symbolic_edges,
            "near_union_edges": near_edges,
            "duplicate_group_count": sum(size > 1 for size in sizes.values()),
            "records_in_duplicate_groups": sum(
                size for size in sizes.values() if size > 1
            ),
            "largest_group_size": max(sizes.values(), default=0),
            "jaccard_threshold": jaccard_threshold,
            "max_hamming_distance": max_hamming_distance,
            "min_near_length": min_near_length,
        },
    )


def _unit_hash(material: str) -> float:
    value = int.from_bytes(
        hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "big"
    )
    return value / 2**64


def assign_leakage_safe_splits(
    records: Sequence[SourceRecord],
    group_ids: Sequence[str],
    *,
    seed: int,
    human_validation_ratio: float = 0.10,
    human_test_ratio: float = 0.10,
) -> tuple[str, ...]:
    """Assign groups atomically; weak AI labels are training-only.

    Any group containing a VOZ-HSD record is forced to train. Consequently,
    validation and test remain human-labelled while cross-source duplicate
    groups cannot leak into evaluation.
    """

    if len(records) != len(group_ids):
        raise ValueError("records and group_ids must have equal length")
    if human_validation_ratio < 0 or human_test_ratio < 0:
        raise ValueError("split ratios cannot be negative")
    if human_validation_ratio + human_test_ratio >= 1:
        raise ValueError("validation + test ratios must be less than 1")

    group_members: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        group_members[group_id].append(index)

    assignments: dict[str, str] = {}
    for group_id, indices in group_members.items():
        if any(records[index].annotation_type != "human" for index in indices):
            assignments[group_id] = "train"
            continue
        label_counts = Counter(records[index].label for index in indices)
        stratum = "+".join(sorted(label_counts))
        value = _unit_hash(f"{seed}\0{stratum}\0{group_id}")
        if value < human_test_ratio:
            assignments[group_id] = "test"
        elif value < human_test_ratio + human_validation_ratio:
            assignments[group_id] = "validation"
        else:
            assignments[group_id] = "train"

    result = tuple(assignments[group_id] for group_id in group_ids)
    assert_no_group_leakage(group_ids, result)
    return result


def assert_no_group_leakage(group_ids: Sequence[str], splits: Sequence[str]) -> None:
    if len(group_ids) != len(splits):
        raise ValueError("group_ids and splits must have equal length")
    observed: dict[str, str] = {}
    for group_id, split in zip(group_ids, splits):
        previous = observed.setdefault(group_id, split)
        if previous != split:
            raise AssertionError(
                f"Duplicate group {group_id} crosses {previous!r} and {split!r}"
            )


__all__ = [
    "DuplicateGroups",
    "assert_no_group_leakage",
    "assign_leakage_safe_splits",
    "group_near_duplicates",
]
