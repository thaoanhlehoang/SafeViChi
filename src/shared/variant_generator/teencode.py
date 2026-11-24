"""Compile a normalization dictionary into a safe generation lexicon."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import unicodedata


TEENCODE_LEXICON_VERSION = "1.0.0"
DEFAULT_TEENCODE_DICT_PATH = (
    Path(__file__).resolve().parents[1] / "normalization" / "teencode_dict.json"
)


class TeencodeLexiconError(ValueError):
    """Raised when the source dictionary cannot be compiled safely."""


@dataclass(frozen=True)
class ExcludedTeencodePair:
    """A source pair withheld from automatic dataset generation."""

    variant: str
    standard: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TeencodeLexicon:
    """Validated, deterministically ordered standard-to-variant mapping."""

    source_path: str
    version: str
    source_sha256: str
    input_pair_count: int
    variants_by_standard: Mapping[str, tuple[str, ...]]
    standard_by_variant: Mapping[str, str]
    excluded_pairs: tuple[ExcludedTeencodePair, ...]

    @property
    def accepted_pair_count(self) -> int:
        return len(self.standard_by_variant)

    @property
    def standard_form_count(self) -> int:
        return len(self.variants_by_standard)

    @property
    def excluded_reason_counts(self) -> dict[str, int]:
        return dict(Counter(pair.reason for pair in self.excluded_pairs))

    def normalizes_pair(self, variant: str, standard: str) -> bool:
        """Return whether an accepted variant maps directly to the standard."""

        return self.standard_by_variant.get(_normalize_lookup(variant)) == (
            _normalize_lookup(standard)
        )

    def rule_id(self, standard: str, variant: str) -> str:
        material = (
            f"{self.version}\0{_normalize_lookup(standard)}\0"
            f"{_normalize_lookup(variant)}"
        )
        suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"teencode_dict:{suffix}"

    def to_manifest(self) -> dict[str, object]:
        return {
            "resource_name": "teencode_dict",
            "resource_version": self.version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "dictionary_direction": "teencode_to_standard",
            "compiled_direction": "standard_to_teencode_variants",
            "input_pair_count": self.input_pair_count,
            "accepted_pair_count": self.accepted_pair_count,
            "standard_form_count": self.standard_form_count,
            "excluded_pair_count": len(self.excluded_pairs),
            "excluded_reason_counts": self.excluded_reason_counts,
        }


def _normalize_surface(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _normalize_lookup(value: str) -> str:
    return _normalize_surface(value).casefold()


def _curated_variant_index(
    curated_mappings: Sequence[Mapping[str, Sequence[str]]],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for mapping in curated_mappings:
        for standard, variants in mapping.items():
            standard_key = _normalize_lookup(standard)
            for variant in variants:
                index[_normalize_lookup(variant)].add(standard_key)
    return dict(index)


def _chain_reason(variant: str, forward: Mapping[str, str]) -> str | None:
    """Classify mappings whose standard output is another dictionary key."""

    current = forward[variant]
    if current not in forward:
        return None
    seen = {variant}
    while current in forward:
        if current in seen:
            return "normalization_cycle"
        seen.add(current)
        current = forward[current]
    return "normalization_chain"


def compile_teencode_lexicon(
    path: str | Path = DEFAULT_TEENCODE_DICT_PATH,
    *,
    curated_mappings: Sequence[Mapping[str, Sequence[str]]] = (),
    version: str = TEENCODE_LEXICON_VERSION,
) -> TeencodeLexicon:
    """Load, validate, deduplicate and reverse a teencode normalization map."""

    source_path = Path(path)
    try:
        payload = source_path.read_bytes()
    except OSError as error:
        raise TeencodeLexiconError(
            f"Cannot read teencode dictionary: {source_path}"
        ) from error
    source_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeencodeLexiconError(
            f"Invalid UTF-8 JSON teencode dictionary: {source_path}"
        ) from error
    if not isinstance(raw, dict):
        raise TeencodeLexiconError("Teencode dictionary root must be a JSON object")

    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    excluded: list[ExcludedTeencodePair] = []
    for raw_variant, raw_standard in raw.items():
        if not isinstance(raw_variant, str) or not isinstance(raw_standard, str):
            raise TeencodeLexiconError("Every teencode dictionary pair must be string-to-string")
        variant = _normalize_surface(raw_variant)
        standard = _normalize_surface(raw_standard)
        if not variant or not standard:
            excluded.append(ExcludedTeencodePair(variant, standard, "empty_form"))
            continue
        if _normalize_lookup(variant) == _normalize_lookup(standard):
            excluded.append(ExcludedTeencodePair(variant, standard, "identity_mapping"))
            continue
        grouped[_normalize_lookup(variant)].append(
            (variant, standard, _normalize_lookup(standard))
        )

    forward: dict[str, str] = {}
    display: dict[str, tuple[str, str]] = {}
    for variant_key, entries in sorted(grouped.items()):
        standards = {standard_key for _, _, standard_key in entries}
        if len(standards) != 1:
            excluded.extend(
                ExcludedTeencodePair(variant, standard, "dictionary_conflict")
                for variant, standard, _ in entries
            )
            continue
        chosen = min(entries, key=lambda entry: (entry[0], entry[1]))
        forward[variant_key] = chosen[2]
        display[variant_key] = (chosen[0], chosen[1])
        for duplicate in entries:
            if duplicate != chosen:
                excluded.append(
                    ExcludedTeencodePair(
                        duplicate[0], duplicate[1], "normalized_duplicate"
                    )
                )

    curated_index = _curated_variant_index(curated_mappings)
    accepted_forward: dict[str, str] = {}
    accepted_display: dict[str, tuple[str, str]] = {}
    for variant_key in sorted(forward):
        standard_key = forward[variant_key]
        variant, standard = display[variant_key]
        chain_reason = _chain_reason(variant_key, forward)
        if chain_reason:
            excluded.append(ExcludedTeencodePair(variant, standard, chain_reason))
            continue
        curated_standards = curated_index.get(variant_key)
        if curated_standards:
            reason = (
                "curated_duplicate"
                if standard_key in curated_standards
                else "curated_conflict"
            )
            excluded.append(ExcludedTeencodePair(variant, standard, reason))
            continue
        accepted_forward[variant_key] = standard_key
        accepted_display[variant_key] = (variant, standard)

    reverse: dict[str, set[str]] = defaultdict(set)
    canonical_standard: dict[str, str] = {}
    standard_by_variant: dict[str, str] = {}
    for variant_key in sorted(accepted_forward):
        variant, standard = accepted_display[variant_key]
        standard_key = accepted_forward[variant_key]
        reverse[standard_key].add(variant)
        canonical_standard.setdefault(standard_key, standard)
        standard_by_variant[variant_key] = standard_key

    variants_by_standard = {
        canonical_standard[standard_key]: tuple(sorted(variants, key=lambda x: x.casefold()))
        for standard_key, variants in sorted(reverse.items())
    }
    return TeencodeLexicon(
        source_path=str(source_path.resolve()),
        version=version,
        source_sha256=source_sha256,
        input_pair_count=len(raw),
        variants_by_standard=variants_by_standard,
        standard_by_variant=standard_by_variant,
        excluded_pairs=tuple(
            sorted(
                excluded,
                key=lambda pair: (
                    pair.reason,
                    pair.variant.casefold(),
                    pair.standard.casefold(),
                ),
            )
        ),
    )


__all__ = [
    "DEFAULT_TEENCODE_DICT_PATH",
    "ExcludedTeencodePair",
    "TEENCODE_LEXICON_VERSION",
    "TeencodeLexicon",
    "TeencodeLexiconError",
    "compile_teencode_lexicon",
]
