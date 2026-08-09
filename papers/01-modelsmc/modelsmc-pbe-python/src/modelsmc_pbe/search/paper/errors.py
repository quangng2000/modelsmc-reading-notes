"""Failures specific to paper-style search orchestration."""


class PaperSearchError(RuntimeError):
    """The practical search could not construct a valid population."""
