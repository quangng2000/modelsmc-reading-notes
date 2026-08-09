"""Finite-skeleton selection at the command boundary."""

from modelsmc_pbe.config import ExperimentConfig
from modelsmc_pbe.domain import ValueType
from modelsmc_pbe.grammar import SkeletonName, available_skeletons


def automatic_skeleton(config: ExperimentConfig) -> SkeletonName:
    """Select the smallest catalog family compatible with the PBE signature."""

    signature = config.spec.signature
    if signature is None:
        raise ValueError("the validated PBE specification has no signature")
    if (
        signature.input_type is ValueType.INT
        and signature.output_type is ValueType.INT
    ):
        return "expression-arithmetic"
    if (
        signature.input_type is ValueType.INT_LIST
        and signature.output_type is ValueType.INT_LIST
    ):
        return "map-arithmetic"
    choices = ", ".join(available_skeletons())
    raise ValueError(
        f"no automatic finite skeleton for {signature.input_type.value} -> "
        f"{signature.output_type.value}; choose one of: {choices}"
    )


def resolve_skeleton(config: ExperimentConfig, requested: str) -> SkeletonName:
    """Resolve ``auto`` or validate an explicitly named finite skeleton."""

    if requested == "auto":
        return automatic_skeleton(config)
    if requested not in available_skeletons():
        choices = ", ".join(available_skeletons())
        raise ValueError(f"unknown skeleton {requested!r}; expected one of: {choices}")
    return requested
