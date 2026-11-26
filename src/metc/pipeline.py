"""End-to-end edit extraction, classification, merging, and rendering."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .errors import MarkerError, PredictionCountError
from .extractor import (
    compute_annotated_edits_result,
    compute_annotated_edits_result_tokenized,
)
from .merge import AnnotationMerger
from .model import DEFAULT_MAX_SEQUENCE_LENGTH, EDIT_CLOSE, EDIT_OPEN, TokenClassifier
from .render import render_classified
from .types import Annotation, ClassificationResult, EditPrediction


class METC:
    """Standalone METC classifier matching the released inference pipeline."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        classifier: TokenClassifier | None = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        device: str = "auto",
        inference_profile: str = "portable",
    ) -> None:
        if classifier is not None and model_path is not None:
            raise ValueError("pass either model_path or classifier, not both")
        self.classifier = (
            classifier
            if classifier is not None
            else TokenClassifier(
                model_path,
                max_sequence_length=max_sequence_length,
                device=device,
                inference_profile=inference_profile,
            )
        )

    @staticmethod
    def _reject_reserved_markers(source: str, target: str) -> None:
        for marker in (EDIT_OPEN, EDIT_CLOSE):
            if marker in source or marker in target:
                raise MarkerError(
                    f"source and target must not contain the reserved marker {marker}"
                )

    @staticmethod
    def _extract(
        source: str,
        target: str,
        *,
        tokenized: bool,
        cleanup_semantic: bool,
    ) -> dict:
        METC._reject_reserved_markers(source, target)
        if tokenized:
            return compute_annotated_edits_result_tokenized(source, target)
        return compute_annotated_edits_result(source, target, cleanup_semantic=cleanup_semantic)

    @staticmethod
    def _build_result(
        source: str,
        extracted: dict,
        predictions: Sequence[EditPrediction],
        *,
        tokenized: bool,
    ) -> ClassificationResult:
        edits = extracted["edits"]
        if len(edits) != len(predictions):
            raise PredictionCountError(
                f"extractor produced {len(edits)} edits but model returned "
                f"{len(predictions)} predictions"
            )
        annotations: list[Annotation] = []
        for edit, prediction in zip(edits, predictions, strict=True):
            if tokenized:
                annotation = Annotation.tokenized(
                    int(edit["startToken"]),
                    int(edit["endToken"]),
                    prediction.label,
                    prediction.sublabel,
                    str(edit["target"]),
                )
            else:
                annotation = Annotation.non_tokenized(
                    int(edit["start"]),
                    int(edit["end"]),
                    prediction.label,
                    prediction.sublabel,
                    str(edit["target"]),
                )
            annotations.append(annotation)

        merged = AnnotationMerger(
            source,
            tokenized=tokenized,
            multilingual=True,
            granular_compound=True,
        ).merge(annotations)
        return ClassificationResult(
            classified=render_classified(source, merged, tokenized=tokenized),
            annotations=tuple(merged),
        )

    def classify(
        self,
        source: str,
        target: str,
        *,
        tokenized: bool = False,
        cleanup_semantic: bool = False,
    ) -> ClassificationResult:
        extracted = self._extract(
            source,
            target,
            tokenized=tokenized,
            cleanup_semantic=cleanup_semantic,
        )
        predictions = (
            self.classifier.predict_marked(
                extracted["annotated"], expected_edit_count=len(extracted["edits"])
            )
            if extracted["edits"]
            else []
        )
        return self._build_result(source, extracted, predictions, tokenized=tokenized)

    def classify_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        tokenized: bool = False,
        cleanup_semantic: bool = False,
    ) -> list[ClassificationResult]:
        extracted = [
            self._extract(
                source,
                target,
                tokenized=tokenized,
                cleanup_semantic=cleanup_semantic,
            )
            for source, target in pairs
        ]
        active_indices = [index for index, item in enumerate(extracted) if item["edits"]]
        predictions: list[list[EditPrediction]] = [[] for _ in pairs]
        if active_indices:
            active_predictions = self.classifier.predict_marked_batch(
                [extracted[index]["annotated"] for index in active_indices],
                [len(extracted[index]["edits"]) for index in active_indices],
            )
            for index, item_predictions in zip(active_indices, active_predictions, strict=True):
                predictions[index] = item_predictions
        return [
            self._build_result(
                source,
                item,
                item_predictions,
                tokenized=tokenized,
            )
            for (source, _), item, item_predictions in zip(
                pairs, extracted, predictions, strict=True
            )
        ]

    def label_annotated(self, annotated: str) -> list[EditPrediction]:
        """Classify an already marked string without extracting or merging edits."""

        return self.classifier.predict_marked(annotated)
