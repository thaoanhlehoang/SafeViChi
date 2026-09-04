"""CLI for the reproducible Vietnamese non-standard text dataset build."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

from src.variant_generator.controlled import (
    BASE_PERTURBATION_TYPES,
    ControlledPerturber,
    DEFAULT_TEENCODE_SPAN_RATE,
    GENERATOR_VERSION,
    PerturbationError,
    load_teencode_lexicon,
)
from src.variant_generator.teencode import TeencodeLexicon

from .constants import (
    DATASET_VERSION,
    LABEL_IDS,
    TARGET_LABEL_COUNTS,
    VIHSD_MAIN_REVISION,
    VIHSD_REPO_ID,
    VOZ_MAIN_REVISION,
    VOZ_PARQUET_REVISION,
    VOZ_REPO_ID,
)
from .dedup import (
    assert_no_group_leakage,
    assign_leakage_safe_splits,
    group_near_duplicates,
    merge_duplicate_groupings,
)
from .quality import (
    run_automated_qa,
    select_human_review_rows,
    write_human_review_csv,
)
from .schema import RejectedRecord, SourceRecord
from .sources import (
    SourceDataError,
    canonical_text,
    iter_voz_hsd,
    load_vihsd_train,
    sample_voz_records,
    sha256_text,
)


KNOWN_OUTPUT_NAMES = {
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "rejected.jsonl",
    "manifest.json",
    "qa_report.json",
    "human_review_sample.csv",
    "teencode_lexicon_audit.json",
    "DATASET_CARD.md",
}


def _derive_seed(master_seed: int, record: SourceRecord, attempt: int) -> int:
    material = (
        f"{master_seed}\0{record.source_dataset}\0{record.source_artifact_revision}\0"
        f"{record.source_row_id}\0{sha256_text(record.text)}\0{attempt}"
    )
    return int.from_bytes(
        hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _sample_id(record: SourceRecord) -> str:
    material = (
        f"{DATASET_VERSION}\0{record.source_dataset}\0{record.source_revision}\0"
        f"{record.source_row_id}\0{sha256_text(record.text)}"
    )
    return "sv_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _preferred_type(
    master_seed: int,
    record: SourceRecord,
    *,
    options: Sequence[str] | None = None,
) -> str:
    options = tuple(options or (*BASE_PERTURBATION_TYPES, "mixed"))
    material = f"{master_seed}\0primary\0{record.source_dataset}\0{record.source_row_id}"
    value = int.from_bytes(
        hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "big"
    )
    return options[value % len(options)]


def _teencode_rank(master_seed: int, record: SourceRecord) -> bytes:
    material = (
        f"{master_seed}\0teencode-quota\0{record.source_dataset}\0"
        f"{record.source_artifact_revision}\0{record.source_row_id}\0"
        f"{sha256_text(record.text)}"
    )
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).digest()


def _plan_teencode_quota(
    records: Sequence[SourceRecord],
    target_splits: Sequence[str],
    *,
    perturber: ControlledPerturber,
    master_seed: int,
    target_rate: float,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Select an exact rounded quota within every source/label/split stratum."""

    if len(records) != len(target_splits):
        raise ValueError("records and target_splits must have equal length")
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError("teencode target rate must be between 0 and 1")

    valid_span_counts = tuple(
        perturber.count_teencode_valid_spans(record.text, record.label)
        for record in records
    )
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (record, split, valid_span_count) in enumerate(
        zip(records, target_splits, valid_span_counts)
    ):
        if valid_span_count:
            strata[(record.source_dataset, record.label, split)].append(index)

    targeted = [False] * len(records)
    for indices in strata.values():
        target_count = int(len(indices) * target_rate + 0.5)
        ordered = sorted(
            indices,
            key=lambda index: (_teencode_rank(master_seed, records[index]), index),
        )
        for index in ordered[:target_count]:
            targeted[index] = True
    return valid_span_counts, tuple(targeted)


def _load_cached_voz_selection(
    path: str | Path,
    *,
    expected_counts: Mapping[str, int],
) -> tuple[SourceRecord, ...]:
    """Recover a previously selected VOZ subset from a generated JSONL split."""

    selection_path = Path(path)
    if not selection_path.is_file():
        raise SourceDataError(f"Cached VOZ selection not found: {selection_path}")
    records: list[SourceRecord] = []
    seen_canonical: set[str] = set()
    with selection_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_dataset") != VOZ_REPO_ID:
                continue
            if row.get("source_revision") != VOZ_MAIN_REVISION:
                raise SourceDataError(
                    f"Cached VOZ source revision mismatch at line {line_number}"
                )
            if row.get("source_artifact_revision") != VOZ_PARQUET_REVISION:
                raise SourceDataError(
                    f"Cached VOZ artifact revision mismatch at line {line_number}"
                )
            text = str(row.get("original_text", ""))
            key = canonical_text(text)
            if not key or key in seen_canonical:
                raise SourceDataError(
                    f"Cached VOZ selection has an empty/duplicate source at line {line_number}"
                )
            seen_canonical.add(key)
            label = str(row.get("label"))
            records.append(
                SourceRecord(
                    source_dataset=VOZ_REPO_ID,
                    source_revision=VOZ_MAIN_REVISION,
                    source_artifact_revision=VOZ_PARQUET_REVISION,
                    source_split=str(row.get("source_split", "train")),
                    source_row_id=str(row["source_row_id"]),
                    text=text,
                    label_original=str(row.get("label_original", label)),
                    label=label,
                    annotation_type="weak_ai",
                    source_confidence=float(row["source_confidence"]),
                    source_artifact=str(row.get("source_artifact") or ""),
                )
            )
    counts = Counter(record.label for record in records)
    if dict(counts) != dict(expected_counts):
        raise SourceDataError(
            f"Cached VOZ selection counts {dict(counts)} do not match {dict(expected_counts)}"
        )
    return tuple(records)


def augment_records(
    records: Sequence[SourceRecord],
    group_ids: Sequence[str],
    target_splits: Sequence[str],
    *,
    master_seed: int,
    max_attempts: int = 36,
    teencode_lexicon: TeencodeLexicon | None = None,
    teencode_rate: float = 1.0,
    teencode_span_rate: float = DEFAULT_TEENCODE_SPAN_RATE,
) -> list[dict[str, object]]:
    """Generate one distinct, traceable variant for every accepted source row."""

    if not (len(records) == len(group_ids) == len(target_splits)):
        raise ValueError("records, group_ids, and target_splits must have equal length")
    if not 0.0 <= teencode_rate <= 1.0:
        raise ValueError("teencode_rate must be between 0 and 1")
    if not 0.0 < teencode_span_rate <= 1.0:
        raise ValueError("teencode_span_rate must be greater than 0 and at most 1")
    lexicon = teencode_lexicon or load_teencode_lexicon()
    perturber = ControlledPerturber(
        teencode_lexicon=lexicon,
        teencode_span_rate=teencode_span_rate,
    )
    teencode_valid_span_counts, teencode_targeted = _plan_teencode_quota(
        records,
        target_splits,
        perturber=perturber,
        master_seed=master_seed,
        target_rate=teencode_rate,
    )
    group_sizes = Counter(group_ids)
    output_owners: dict[str, tuple[str, str]] = {}
    rows: list[dict[str, object]] = []
    non_teencode_options = tuple(
        perturbation_type
        for perturbation_type in BASE_PERTURBATION_TYPES
        if perturbation_type != "teencode_lexical"
    ) + ("mixed",)

    for index, (record, group_id, target_split) in enumerate(
        zip(records, group_ids, target_splits)
    ):
        is_teencode_target = teencode_targeted[index]
        preferred = (
            "teencode_lexical"
            if is_teencode_target
            else _preferred_type(
                master_seed, record, options=non_teencode_options
            )
        )
        preferred_index = (
            0 if is_teencode_target else non_teencode_options.index(preferred)
        )
        result = None
        duplicate_fallback = None
        for attempt in range(max_attempts):
            required_type = (
                "teencode_lexical"
                if is_teencode_target
                else non_teencode_options[
                    (preferred_index + attempt) % len(non_teencode_options)
                ]
            )
            seed = _derive_seed(master_seed, record, attempt)
            try:
                candidate = perturber.perturb(
                    record.text,
                    record.label,
                    seed=seed,
                    required_type=required_type,
                    min_operations=2 if required_type == "mixed" else 1,
                    max_operations=3,
                    disabled_types=(
                        () if is_teencode_target else ("teencode_lexical",)
                    ),
                    teencode_span_rate=teencode_span_rate,
                )
            except PerturbationError:
                continue
            output_key = canonical_text(candidate.perturbed_text)
            if not output_key:
                continue
            owner = output_owners.get(output_key)
            if owner is not None:
                if owner[1] == target_split and duplicate_fallback is None:
                    duplicate_fallback = candidate
                continue
            result = candidate
            output_owners[output_key] = (group_id, target_split)
            break
        used_duplicate_fallback = result is None and duplicate_fallback is not None
        if used_duplicate_fallback:
            result = duplicate_fallback
        if result is None:
            raise SourceDataError(
                "Could not create a distinct safe variant after "
                f"{max_attempts} attempts for {record.source_dataset} row "
                f"{record.source_row_id}."
            )

        teencode_applied = "teencode_lexical" in result.perturbation_types
        teencode_applied_span_count = sum(
            perturbation_type == "teencode_lexical"
            for perturbation_type in result.perturbation_types
        )
        valid_span_count = teencode_valid_span_counts[index]
        expected_teencode_span_count = (
            max(1, math.ceil(valid_span_count * teencode_span_rate))
            if is_teencode_target
            else 0
        )
        if teencode_applied != is_teencode_target:
            raise AssertionError(
                "Internal error: teencode application differs from its quota plan"
            )
        if teencode_applied_span_count != expected_teencode_span_count:
            raise AssertionError(
                "Internal error: teencode span application differs from its target"
            )

        flags: list[str] = []
        if record.annotation_type == "weak_ai":
            flags.append("weak_ai_source_label")
        if group_sizes[group_id] > 1:
            flags.append("source_duplicate_or_near_duplicate_group")
        if used_duplicate_fallback:
            flags.append("perturbed_output_duplicate_same_split")

        metadata = result.to_metadata()
        # A fixed-width hexadecimal string is exact in JSON/JavaScript and
        # avoids signed-int overflow during Arrow schema inference.
        metadata["random_seed"] = f"{result.seed:016x}"
        row: dict[str, object] = {
            "sample_id": _sample_id(record),
            "text": result.perturbed_text,
            "original_text": record.text,
            "label": record.label,
            "label_mapped": record.label,
            "label_id": LABEL_IDS[record.label],
            "label_original": record.label_original,
            "source_dataset": record.source_dataset,
            "source_revision": record.source_revision,
            "source_artifact_revision": record.source_artifact_revision,
            "source_artifact": record.source_artifact,
            "source_split": record.source_split,
            "source_row_id": record.source_row_id,
            "source_confidence": record.source_confidence,
            "annotation_type": record.annotation_type,
            "target_split": target_split,
            "duplicate_group_id": group_id,
            "duplicate_group_size": group_sizes[group_id],
            "original_text_sha256": sha256_text(record.text),
            "perturbed_text_sha256": sha256_text(result.perturbed_text),
            "perturbation_primary_requested": preferred,
            **metadata,
            "teencode_eligible": valid_span_count > 0,
            "teencode_targeted": is_teencode_target,
            "teencode_applied": teencode_applied,
            "teencode_valid_span_count": valid_span_count,
            "teencode_applied_span_count": teencode_applied_span_count,
            "teencode_span_target_rate": teencode_span_rate,
            "dataset_version": DATASET_VERSION,
            "quality_flags": flags,
        }
        rows.append(row)
        if (index + 1) % 10_000 == 0:
            print(
                f"Augmented {index + 1:,}/{len(records):,} accepted source rows",
                flush=True,
            )
    return rows


def _with_progress(
    records: Iterable[SourceRecord], *, every: int = 1_000_000
) -> Iterator[SourceRecord]:
    for index, record in enumerate(records, start=1):
        if index % every == 0:
            print(f"Scanned {index:,} VOZ-HSD rows", flush=True)
        yield record


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SourceDataError(
            "Parquet output requires pyarrow; install requirements.txt or pass --jsonl-only."
        ) from error

    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(list(rows))
        metadata = dict(table.schema.metadata or {})
        metadata[b"safevichi_dataset_version"] = DATASET_VERSION.encode("ascii")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_fingerprint(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        for field in (
            "sample_id",
            "perturbed_text_sha256",
            "label",
            "target_split",
            "random_seed",
            "generator_version",
        ):
            digest.update(str(row[field]).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _dataset_card(manifest: Mapping[str, object]) -> str:
    final_counts = manifest["final_counts"]
    teencode = manifest["teencode_augmentation"]
    teencode_resource = manifest["perturbation_resources"]["teencode_dict"]
    return f"""---
language:
- vi
task_categories:
- text-classification
pretty_name: SafeViChi Vietnamese Non-standard / Perturbed Text Dataset
---

# SafeViChi Vietnamese Non-standard / Perturbed Text Dataset

Version: `{DATASET_VERSION}`
Fingerprint: `{manifest['dataset_fingerprint']}`

This build contains {final_counts['total']:,} controlled surface variants for
binary `HATE`/`CLEAN` classification. Every accepted row contains its original
text, perturbed text, source/revision/row provenance, mapped and original label,
sequential edit trace, per-row random seed, duplicate group, and generator
version.

The `teencode_lexical` transform was applied to
{teencode['applied_count']:,}/{teencode['eligible_count']:,} eligible rows
({teencode['applied_among_eligible_rate']:.2%}), using the validated
`teencode_dict` resource at SHA-256 `{teencode_resource['source_sha256']}`.
It replaced {teencode['applied_span_count']:,} of
{teencode['valid_span_count_in_targeted_rows']:,} valid spans in targeted rows
({teencode['span_application_rate']:.2%}). Rows without a valid dictionary span
are retained and explicitly marked ineligible rather than receiving fabricated
content.
The compiler accepted {teencode_resource['accepted_pair_count']:,} of
{teencode_resource['input_pair_count']:,} source pairs; exclusion counts are
recorded in `manifest.json` and pair-level details in
`teencode_lexicon_audit.json`.

## Sources and label provenance

- [`uitnlp/vihsd`](https://huggingface.co/datasets/uitnlp/vihsd), pinned at
  `{VIHSD_MAIN_REVISION}`: human labels; `OFFENSIVE` and `HATE` map to `HATE`.
- [`tarudesu/VOZ-HSD`](https://huggingface.co/datasets/tarudesu/VOZ-HSD), pinned
  at source revision `{VOZ_MAIN_REVISION}` and converted-Parquet revision
  `{VOZ_PARQUET_REVISION}`: weak AI-generated labels (`1=HATE`, `0=CLEAN`).

VOZ-HSD's authors state that the dataset is for research and that its model-
generated labels were intended for analysis, not downstream fine-tuning.
Accordingly, these rows are marked `annotation_type=weak_ai`, forced into the
training split, and must not be treated as gold benchmark labels. Obtain any
needed permission and human re-annotation before public release or downstream
training. Validation and test contain human-labelled ViHSD rows only.

## Quality controls

- Empty texts and exact duplicate groups with contradictory binary ViHSD labels
  are quarantined in `rejected.jsonl` (hashes and reasons only).
- Exact and conservative near duplicates share one `duplicate_group_id` and are
  assigned atomically to one target split.
- Every perturbation is local, replayable, seeded, and automatically checked for
  structural label-preservation invariants.
- `human_review_sample.csv` must be completed to assess naturalness, semantic
  preservation, and label preservation; automatic checks do not replace review.

## Primary fields

`text`, `original_text`, `label`, `label_original`, `annotation_type`,
`source_dataset`, `source_revision`, `source_row_id`, `target_split`,
`perturbation_types`, `perturbation_edits`, `perturbation_count`, `random_seed`,
`generator_version`, `teencode_eligible`, `teencode_targeted`,
`teencode_applied`, `teencode_valid_span_count`,
`teencode_applied_span_count`, `teencode_span_target_rate`,
`duplicate_group_id`, and `quality_flags`.
"""


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in output_dir.iterdir() if path.name in KNOWN_OUTPUT_NAMES)
    if existing and not overwrite:
        raise SourceDataError(
            f"Output directory already contains build artifacts: {existing}. "
            "Use --overwrite to replace only these known files."
        )


def write_artifacts(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    rejected: Sequence[RejectedRecord],
    manifest: dict[str, object],
    qa_report: Mapping[str, object],
    teencode_lexicon: TeencodeLexicon,
    *,
    review_size: int,
    seed: int,
    jsonl_only: bool,
    overwrite: bool,
) -> None:
    _prepare_output(output_dir, overwrite=overwrite)
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["target_split"] == split]
        _atomic_write_jsonl(output_dir / f"{split}.jsonl", split_rows)
        if not jsonl_only:
            _atomic_write_parquet(output_dir / f"{split}.parquet", split_rows)
    _atomic_write_jsonl(
        output_dir / "rejected.jsonl", (record.to_dict() for record in rejected)
    )
    _atomic_write_json(output_dir / "qa_report.json", qa_report)
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_json(
        output_dir / "teencode_lexicon_audit.json",
        {
            "summary": teencode_lexicon.to_manifest(),
            "excluded_pairs": [
                pair.to_dict() for pair in teencode_lexicon.excluded_pairs
            ],
        },
    )
    _atomic_write_text(output_dir / "DATASET_CARD.md", _dataset_card(manifest))

    review_rows = select_human_review_rows(rows, size=review_size, seed=seed)
    descriptor, temporary_name = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_human_review_csv(temporary, review_rows)
        os.replace(temporary, output_dir / "human_review_sample.csv")
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    _prepare_output(output_dir, overwrite=args.overwrite)
    if args.voz_selection_jsonl and args.voz_parquet:
        raise SourceDataError(
            "Use either --voz-selection-jsonl or --voz-parquet, not both."
        )
    if args.voz_max_rows < 0:
        raise SourceDataError("--voz-max-rows cannot be negative")
    if args.human_review_size < 0:
        raise SourceDataError("--human-review-size cannot be negative")
    if not 0.0 <= args.teencode_rate <= 1.0:
        raise SourceDataError("--teencode-rate must be between 0 and 1")
    if not 0.0 < args.teencode_span_rate <= 1.0:
        raise SourceDataError(
            "--teencode-span-rate must be greater than 0 and at most 1"
        )

    teencode_lexicon = load_teencode_lexicon(args.teencode_dict)

    vihsd = load_vihsd_train(args.vihsd_csv, conflict_policy=args.conflict_policy)
    print(
        "Loaded ViHSD train: "
        f"{sum(vihsd.mapped_label_counts.values()):,} raw, "
        f"{len(vihsd.records):,} accepted, {len(vihsd.rejected):,} quarantined",
        flush=True,
    )

    target_counts = {"HATE": args.target_hate, "CLEAN": args.target_clean}
    voz_targets = {
        label: target_counts[label] - vihsd.valid_label_counts.get(label, 0)
        for label in target_counts
    }
    if any(count < 0 for count in voz_targets.values()):
        raise SourceDataError(
            f"Final target counts are smaller than accepted ViHSD counts: {voz_targets}"
        )
    excluded = {canonical_text(record.text): record.label for record in vihsd.records}
    if args.voz_selection_jsonl:
        voz_records = _load_cached_voz_selection(
            args.voz_selection_jsonl, expected_counts=voz_targets
        )
        overlap = [
            record.source_row_id
            for record in voz_records
            if canonical_text(record.text) in excluded
        ]
        if overlap:
            raise SourceDataError(
                f"Cached VOZ selection overlaps ViHSD ({len(overlap)} exact duplicates)"
            )
        below_threshold = sum(
            record.source_confidence is None
            or record.source_confidence < args.voz_min_confidence
            for record in voz_records
        )
        if below_threshold:
            raise SourceDataError(
                f"Cached VOZ selection has {below_threshold} rows below confidence threshold"
            )
        voz_stats: Mapping[str, object] = {
            "sampling_method": (
                "stratified_deterministic_priority_reservoir_recovered_from_jsonl"
            ),
            "selection_jsonl": str(args.voz_selection_jsonl),
            "scan_complete": None,
            "selected_by_label": dict(Counter(record.label for record in voz_records)),
            "min_confidence": args.voz_min_confidence,
            "minimum_selected_confidence": min(
                record.source_confidence for record in voz_records if record.source_confidence
            ),
        }
    else:
        voz_locations = args.voz_parquet or None
        voz_stream = _with_progress(iter_voz_hsd(voz_locations))
        voz = sample_voz_records(
            voz_stream,
            voz_targets,
            seed=args.seed,
            min_confidence=args.voz_min_confidence,
            oversample_factor=args.oversample_factor,
            max_rows=args.voz_max_rows or None,
            exclude_canonical=excluded,
        )
        voz_records = voz.records
        voz_stats = voz.stats
    print(
        "Selected VOZ-HSD weak labels: "
        + ", ".join(
            f"{label}={count:,}"
            for label, count in sorted(Counter(r.label for r in voz_records).items())
        ),
        flush=True,
    )

    records = (*vihsd.records, *voz_records)
    if len(records) != sum(target_counts.values()):
        raise AssertionError("Internal error: accepted source count misses final target")
    duplicate_groups = group_near_duplicates(
        records,
        jaccard_threshold=args.near_duplicate_threshold,
        max_hamming_distance=args.near_duplicate_hamming,
        min_near_length=args.near_duplicate_min_length,
    )
    target_splits = assign_leakage_safe_splits(
        records,
        duplicate_groups.group_ids,
        seed=args.seed,
        human_validation_ratio=args.validation_ratio,
        human_test_ratio=args.test_ratio,
    )
    rows = augment_records(
        records,
        duplicate_groups.group_ids,
        target_splits,
        master_seed=args.seed,
        teencode_lexicon=teencode_lexicon,
        teencode_rate=args.teencode_rate,
        teencode_span_rate=args.teencode_span_rate,
    )
    perturbed_records = tuple(
        replace(record, text=str(row["text"])) for record, row in zip(records, rows)
    )
    perturbed_duplicate_groups = group_near_duplicates(
        perturbed_records,
        jaccard_threshold=args.near_duplicate_threshold,
        max_hamming_distance=args.near_duplicate_hamming,
        min_near_length=args.near_duplicate_min_length,
    )
    combined_duplicate_groups = merge_duplicate_groupings(
        duplicate_groups.group_ids,
        perturbed_duplicate_groups.group_ids,
    )
    target_splits = assign_leakage_safe_splits(
        records,
        combined_duplicate_groups.group_ids,
        seed=args.seed,
        human_validation_ratio=args.validation_ratio,
        human_test_ratio=args.test_ratio,
    )
    combined_group_sizes = Counter(combined_duplicate_groups.group_ids)
    for row, group_id, target_split in zip(
        rows,
        combined_duplicate_groups.group_ids,
        target_splits,
    ):
        row["duplicate_group_id"] = group_id
        row["duplicate_group_size"] = combined_group_sizes[group_id]
        row["target_split"] = target_split

    assert_no_group_leakage(duplicate_groups.group_ids, target_splits)
    assert_no_group_leakage(perturbed_duplicate_groups.group_ids, target_splits)
    assert_no_group_leakage(combined_duplicate_groups.group_ids, target_splits)
    qa_report = run_automated_qa(
        rows,
        target_label_counts=target_counts,
        teencode_lexicon=teencode_lexicon,
        teencode_target_rate=args.teencode_rate,
        teencode_span_target_rate=args.teencode_span_rate,
    )
    qa_report["perturbed_duplicate_grouping"] = dict(
        perturbed_duplicate_groups.stats
    )
    qa_report["combined_duplicate_grouping"] = dict(
        combined_duplicate_groups.stats
    )
    qa_report["source_near_duplicate_groups_crossing_splits"] = 0
    qa_report["perturbed_near_duplicate_groups_crossing_splits"] = 0
    qa_report["combined_near_duplicate_groups_crossing_splits"] = 0

    final_by_source = Counter(record.source_dataset for record in records)
    final_by_label = Counter(record.label for record in records)
    final_by_split = Counter(target_splits)
    fingerprint = _dataset_fingerprint(rows)
    manifest: dict[str, object] = {
        "dataset_name": "safevichi-vietnamese-nonstandard",
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "random_seed": args.seed,
        "final_counts": {
            "total": len(records),
            "by_label": dict(final_by_label),
            "by_source": dict(final_by_source),
            "by_split": dict(final_by_split),
        },
        "requested_target_counts": target_counts,
        "sources": {
            VIHSD_REPO_ID: {
                "revision": VIHSD_MAIN_REVISION,
                "source_split": "train",
                "schema": ["free_text", "label_id"],
                "raw_label_counts": dict(vihsd.raw_label_counts),
                "mapped_label_counts": dict(vihsd.mapped_label_counts),
                "accepted_label_counts": dict(vihsd.valid_label_counts),
                "quarantined_count": len(vihsd.rejected),
                "conflict_policy": args.conflict_policy,
                "annotation_type": "human",
            },
            VOZ_REPO_ID: {
                "revision": VOZ_MAIN_REVISION,
                "artifact_revision": VOZ_PARQUET_REVISION,
                "source_split": "train",
                "schema": ["texts", "labels", "probs"],
                "requested_label_counts_after_vihsd_quarantine": voz_targets,
                "annotation_type": "weak_ai",
                "evaluation_eligible": False,
                "usage_notice": (
                    "Research-only source; authors describe model-generated labels as "
                    "analysis labels, not downstream fine-tuning labels."
                ),
                "sampling": dict(voz_stats),
            },
        },
        "label_mapping": {
            "ViHSD": {"HATE": "HATE", "OFFENSIVE": "HATE", "CLEAN": "CLEAN"},
            "VOZ-HSD": {"1": "HATE", "0": "CLEAN"},
        },
        "perturbation_resources": {
            "teencode_dict": teencode_lexicon.to_manifest(),
        },
        "teencode_augmentation": qa_report["teencode_lexical"],
        "split_policy": {
            "method": (
                "transitive source-and-perturbed duplicate-group atomic "
                "deterministic hashing"
            ),
            "human_validation_ratio": args.validation_ratio,
            "human_test_ratio": args.test_ratio,
            "weak_ai_rows": "train_only",
        },
        "duplicate_grouping": dict(duplicate_groups.stats),
        "perturbed_duplicate_grouping": dict(perturbed_duplicate_groups.stats),
        "combined_duplicate_grouping": dict(combined_duplicate_groups.stats),
        "quality_gate_status": qa_report["status"],
        "human_review_required": True,
        "distribution_status": "rights_and_weak_label_review_required_before_release",
        "build_parameters": {
            "vihsd_csv": str(args.vihsd_csv),
            "voz_parquet": list(args.voz_parquet or []),
            "voz_selection_jsonl": args.voz_selection_jsonl,
            "voz_min_confidence": args.voz_min_confidence,
            "voz_max_rows": args.voz_max_rows or None,
            "oversample_factor": args.oversample_factor,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "near_duplicate_hamming": args.near_duplicate_hamming,
            "near_duplicate_min_length": args.near_duplicate_min_length,
            "teencode_dict": str(args.teencode_dict),
            "teencode_rate": args.teencode_rate,
            "teencode_span_rate": args.teencode_span_rate,
        },
    }
    write_artifacts(
        output_dir,
        rows,
        vihsd.rejected,
        manifest,
        qa_report,
        teencode_lexicon,
        review_size=args.human_review_size,
        seed=args.seed,
        jsonl_only=args.jsonl_only,
        overwrite=args.overwrite,
    )
    print(
        f"Build passed QA: {len(rows):,} rows, fingerprint={fingerprint}",
        flush=True,
    )
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the reproducible SafeViChi Vietnamese perturbed-text corpus"
    )
    parser.add_argument("--vihsd-csv", default="data/raw/vihsd/train.csv")
    parser.add_argument(
        "--voz-parquet",
        nargs="*",
        help="Optional local VOZ-HSD Parquet shards; pinned remote shards are the default",
    )
    parser.add_argument(
        "--voz-selection-jsonl",
        help=(
            "Reuse validated VOZ rows from an earlier generated train JSONL; "
            "skips the 10.7M-row source scan"
        ),
    )
    parser.add_argument(
        "--output-dir", default="data/processed/vietnamese_nonstandard_v1_2"
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--target-hate", type=int, default=TARGET_LABEL_COUNTS["HATE"])
    parser.add_argument("--target-clean", type=int, default=TARGET_LABEL_COUNTS["CLEAN"])
    parser.add_argument("--voz-min-confidence", type=float, default=0.90)
    parser.add_argument("--voz-max-rows", type=int, default=0)
    parser.add_argument("--oversample-factor", type=float, default=1.25)
    parser.add_argument(
        "--conflict-policy", choices=("quarantine", "keep"), default="quarantine"
    )
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--near-duplicate-hamming", type=int, default=8)
    parser.add_argument("--near-duplicate-min-length", type=int, default=12)
    parser.add_argument("--human-review-size", type=int, default=250)
    parser.add_argument(
        "--teencode-dict",
        default="src/normalization/teencode_dict.json",
        help="Normalization dictionary compiled into generation rules",
    )
    parser.add_argument(
        "--teencode-rate",
        type=float,
        default=1.0,
        help="Rounded per-stratum share of eligible rows requiring teencode_lexical",
    )
    parser.add_argument(
        "--teencode-span-rate",
        type=float,
        default=DEFAULT_TEENCODE_SPAN_RATE,
        help="Minimum per-sample share of valid dictionary spans to replace",
    )
    parser.add_argument("--jsonl-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
