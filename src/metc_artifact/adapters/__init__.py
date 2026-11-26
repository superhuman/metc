"""Subprocess-isolated producers for freshly regenerated annotations."""

from .contract import (
    AnnotationRecord,
    NormalizedEdit,
    PairRecord,
    materialize_paper_rows,
    pairs_from_paper_rows,
    read_annotations,
    read_pairs,
    write_annotations,
    write_pairs,
)
from .errors import (
    AdapterError,
    AdapterInputError,
    AdapterOutputError,
    AdapterProcessError,
    AdapterUnavailableError,
)

__all__ = [
    "AdapterError",
    "AdapterInputError",
    "AdapterOutputError",
    "AdapterProcessError",
    "AdapterUnavailableError",
    "AnnotationRecord",
    "NormalizedEdit",
    "PairRecord",
    "materialize_paper_rows",
    "pairs_from_paper_rows",
    "read_annotations",
    "read_pairs",
    "write_annotations",
    "write_pairs",
]
