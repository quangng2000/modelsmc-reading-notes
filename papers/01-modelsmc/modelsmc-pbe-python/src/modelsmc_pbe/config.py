"""Validated experiment configuration for flat and nested JSON inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    model_validator,
)

from modelsmc_pbe.domain.models import PBESpec


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
FinitePositiveFloat = Annotated[
    float, Field(strict=True, gt=0.0, allow_inf_nan=False)
]
FiniteNonNegativeFloat = Annotated[
    float, Field(strict=True, ge=0.0, allow_inf_nan=False)
]
Probability = Annotated[
    float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
]


class SMCConfig(_ConfigModel):
    """Algorithm settings shared by LLM-guided and grammar-based SMC runs."""

    particles: PositiveInteger = 8
    iterations: Annotated[int, Field(strict=True, gt=0, le=10_000)] = 6
    clone_probability: Probability = Field(default=0.35, alias="cloneProbability")
    ess_threshold: Annotated[
        float, Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False)
    ] = Field(default=0.6, alias="essThreshold")
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**63)] = 7
    loss_scale: FinitePositiveFloat = Field(default=2.0, alias="lossScale")
    cost_scale: FiniteNonNegativeFloat = Field(default=0.15, alias="costScale")
    loss_cap: Annotated[int, Field(strict=True, gt=0, le=2**53 - 1)] = Field(
        default=1_000_000, alias="lossCap"
    )
    max_cost: Annotated[int, Field(strict=True, gt=0, le=2**53 - 1)] = Field(
        default=20, alias="maxCost"
    )
    max_depth: Annotated[int, Field(strict=True, gt=0, le=64)] = Field(
        default=10, alias="maxDepth"
    )
    max_nodes: Annotated[int, Field(strict=True, gt=0, le=10_000)] = Field(
        default=127, alias="maxNodes"
    )

    @property
    def alpha(self) -> float:
        """Paper-compatible name for the probability of cloning an ancestor."""

        return self.clone_probability


class RuntimeConfig(_ConfigModel):
    """Hardware, determinism, and artifact-output settings."""

    device: str = "auto"
    deterministic: StrictBool = True
    artifacts_dir: Path = Field(default=Path("runs"), alias="artifactsDir")
    console_level: Literal["trace", "debug", "info", "warning", "error", "quiet"] = Field(
        default="info", alias="consoleLevel"
    )


_SPEC_KEYS = {"name", "signature", "examples", "integerConstants", "integer_constants"}
_SMC_KEYS = {
    "particles",
    "iterations",
    "cloneProbability",
    "clone_probability",
    "alpha",
    "essThreshold",
    "ess_threshold",
    "seed",
    "lossScale",
    "loss_scale",
    "costScale",
    "cost_scale",
    "lossCap",
    "loss_cap",
    "maxCost",
    "max_cost",
    "maxDepth",
    "max_depth",
    "maxNodes",
    "max_nodes",
}
_RUNTIME_KEYS = {
    "device",
    "deterministic",
    "artifactsDir",
    "artifacts_dir",
    "consoleLevel",
    "console_level",
}


class ExperimentConfig(_ConfigModel):
    """One normalized experiment assembled from spec, SMC, and runtime data.

    Both the nested experiment form and the compact flat JSON form are accepted. The
    normalized in-memory representation is always nested.
    """

    spec: PBESpec
    smc: SMCConfig = Field(default_factory=SMCConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="before")
    @classmethod
    def split_flat_config(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "spec" in value:
            return value

        unknown = set(value) - _SPEC_KEYS - _SMC_KEYS - _RUNTIME_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown experiment configuration field(s): {names}")

        spec = {key: item for key, item in value.items() if key in _SPEC_KEYS}
        smc = {key: item for key, item in value.items() if key in _SMC_KEYS}
        runtime = {key: item for key, item in value.items() if key in _RUNTIME_KEYS}
        if "alpha" in smc:
            if "cloneProbability" in smc or "clone_probability" in smc:
                raise ValueError("specify only one of alpha and cloneProbability")
            smc["cloneProbability"] = smc.pop("alpha")
        return {"spec": spec, "smc": smc, "runtime": runtime}


def load_experiment_config(
    path: str | Path,
    *,
    smc_overrides: dict[str, Any] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load, override, and validate an experiment JSON document."""

    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"experiment configuration does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"experiment configuration must be a JSON object: {source}")

    config = ExperimentConfig.model_validate(raw)
    if smc_overrides:
        merged_smc = config.smc.model_dump()
        merged_smc.update(smc_overrides)
        config = config.model_copy(update={"smc": SMCConfig.model_validate(merged_smc)})
    if runtime_overrides:
        merged_runtime = config.runtime.model_dump()
        merged_runtime.update(runtime_overrides)
        config = config.model_copy(update={"runtime": RuntimeConfig.model_validate(merged_runtime)})
    return config
