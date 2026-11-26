"""Public exceptions raised by the standalone METC pipeline."""


class METCError(Exception):
    """Base class for METC release-package errors."""


class CheckpointContractError(METCError):
    """The supplied checkpoint is incompatible with the released pipeline."""


class MarkerError(METCError):
    """Edit markers are malformed or collide with user text."""


class TruncationError(METCError):
    """Reserved compatibility error for callers that reject truncated inputs."""


class PredictionCountError(METCError):
    """The model returned a different number of labels than edits."""
