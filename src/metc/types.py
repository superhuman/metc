"""Stable public data types used by METC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EditPrediction:
    """Top-1 classifier output at one ``[EDIT]`` marker."""

    label_id: int
    label: str
    sublabel: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Annotation:
    """One classified source edit.

    Character offsets use UTF-16 code units, matching the reference
    implementation. Token offsets are used only in tokenized mode.
    """

    label: str
    sublabel: str
    replacement: str
    start: int | None = None
    end: int | None = None
    start_token: int | None = None
    end_token: int | None = None

    @classmethod
    def non_tokenized(
        cls,
        start: int,
        end: int,
        label: str,
        sublabel: str,
        replacement: str,
    ) -> "Annotation":
        return cls(
            start=start,
            end=end,
            label=label,
            sublabel=sublabel,
            replacement=replacement,
        )

    @classmethod
    def tokenized(
        cls,
        start_token: int,
        end_token: int,
        label: str,
        sublabel: str,
        replacement: str,
    ) -> "Annotation":
        return cls(
            start_token=start_token,
            end_token=end_token,
            label=label,
            sublabel=sublabel,
            replacement=replacement,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label,
            "sublabel": self.sublabel,
            "replacement": self.replacement,
        }
        if self.start is not None:
            result["start"] = self.start
            result["end"] = self.end
        if self.start_token is not None:
            result["startToken"] = self.start_token
            result["endToken"] = self.end_token
        return result


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Complete standalone METC response for a source/target pair."""

    classified: str
    annotations: tuple[Annotation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classified": self.classified,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }
