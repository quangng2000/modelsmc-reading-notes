"""Explicit Torch device resolution for reproducible research runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class DeviceResolutionError(RuntimeError):
    """Raised when an explicitly requested accelerator cannot be used."""


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved device plus enough context to record the decision."""

    requested: str
    resolved: str
    accelerator: bool
    reason: str

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.resolved)

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "accelerator": self.accelerator,
            "reason": self.reason,
        }


def _mps_built() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built())


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _cuda_device(requested: str) -> DeviceInfo:
    if not torch.cuda.is_available():
        raise DeviceResolutionError(
            f"device {requested!r} was requested, but torch.cuda.is_available() is false; "
            "install a CUDA-enabled Torch build and verify the NVIDIA driver"
        )
    try:
        parsed = torch.device(requested)
    except RuntimeError as error:
        raise DeviceResolutionError(f"invalid CUDA device {requested!r}: {error}") from error
    index = parsed.index if parsed.index is not None else 0
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise DeviceResolutionError(
            f"device {requested!r} does not exist; Torch reports {count} CUDA device(s)"
        )
    resolved = f"cuda:{index}"
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:  # pragma: no cover - unusual driver failures are still reported cleanly
        name = "CUDA accelerator"
    return DeviceInfo(requested=requested, resolved=resolved, accelerator=True, reason=name)


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve ``auto`` as CUDA, then MPS, then CPU.

    Explicit accelerator requests never fall back silently.  A misspelled or
    unavailable device is a configuration error, which prevents an expensive
    experiment from unexpectedly running on the CPU.
    """

    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            resolved = _cuda_device("cuda")
            return DeviceInfo(
                requested="auto",
                resolved=resolved.resolved,
                accelerator=True,
                reason=f"auto-selected CUDA ({resolved.reason})",
            )
        if _mps_available():
            return DeviceInfo(
                requested="auto",
                resolved="mps",
                accelerator=True,
                reason="auto-selected Apple Metal Performance Shaders",
            )
        return DeviceInfo(
            requested="auto",
            resolved="cpu",
            accelerator=False,
            reason="no CUDA or MPS accelerator is available",
        )
    if normalized == "cpu":
        return DeviceInfo(
            requested=requested,
            resolved="cpu",
            accelerator=False,
            reason="CPU explicitly requested",
        )
    if normalized == "mps":
        if not _mps_built():
            raise DeviceResolutionError(
                "device 'mps' was requested, but this Torch build has no MPS support"
            )
        if not _mps_available():
            raise DeviceResolutionError(
                "device 'mps' was requested, but the current macOS/hardware runtime cannot use MPS"
            )
        return DeviceInfo(
            requested=requested,
            resolved="mps",
            accelerator=True,
            reason="Apple MPS explicitly requested",
        )
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return _cuda_device(normalized)
    raise DeviceResolutionError(
        f"unsupported device {requested!r}; expected auto, cpu, mps, cuda, or cuda:<index>"
    )
