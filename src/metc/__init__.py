"""Public API for the standalone Multilingual Error Type Classification model."""

from .errors import (
    CheckpointContractError,
    MarkerError,
    METCError,
    PredictionCountError,
    TruncationError,
)
from .model import DEFAULT_MAX_SEQUENCE_LENGTH, INFERENCE_PROFILES, TokenClassifier
from .pipeline import METC
from .types import Annotation, ClassificationResult, EditPrediction

__all__ = [
    "Annotation",
    "CheckpointContractError",
    "ClassificationResult",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "EditPrediction",
    "INFERENCE_PROFILES",
    "MarkerError",
    "METC",
    "METCError",
    "PredictionCountError",
    "TokenClassifier",
    "TruncationError",
]
