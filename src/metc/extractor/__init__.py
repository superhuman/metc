"""Language-agnostic edit extraction used to create classifier inputs."""

from .segments import (
    compute_annotated_edits_result,
    compute_annotated_edits_result_tokenized,
)

__all__ = [
    "compute_annotated_edits_result",
    "compute_annotated_edits_result_tokenized",
]
