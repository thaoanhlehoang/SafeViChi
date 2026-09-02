"""Internal source and rejection records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SourceRecord:
    source_dataset: str
    source_revision: str
    source_artifact_revision: str
    source_split: str
    source_row_id: str
    text: str
    label_original: str
    label: str
    annotation_type: str
    source_confidence: float | None = None
    source_artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectedRecord:
    source_dataset: str
    source_split: str
    source_row_id: str
    label_original: str
    label: str
    text_sha256: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
