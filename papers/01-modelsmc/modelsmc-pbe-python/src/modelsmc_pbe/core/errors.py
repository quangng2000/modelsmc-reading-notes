"""Errors that distinguish rejected candidates from broken core invariants."""


class CoreInvariantError(RuntimeError):
    """Raised only when validated, well-typed input violates a core invariant."""
