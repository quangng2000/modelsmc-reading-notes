"""Optional structured event emission shared by paper-search components."""

from modelsmc_pbe.observability import RunLogger


def emit(
    logger: RunLogger | None,
    name: str,
    *,
    message: str,
    level: str = "info",
    **data: object,
) -> None:
    """Emit an event when the caller supplied a run logger."""

    if logger is not None:
        logger.event(name, message=message, level=level, **data)
