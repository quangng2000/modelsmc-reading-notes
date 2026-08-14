"""JSON boundary rules shared by every research-run artifact."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

_SECRET_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "access_token",
)
_RAW_PAYLOAD_KEYS = {
    "prompt",
    "rationale",
    "raw_prompt",
    "raw_response",
    "response",
}


def iso_now() -> str:
    """Return the current UTC time in the artifact schema's stable format."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def jsonable(value: Any, *, include_raw_payloads: bool, key: str | None = None) -> Any:
    """Convert an object to safe JSON data, redacting secrets and raw prompts."""

    normalized_key = key.lower() if key else ""
    if any(part in normalized_key for part in _SECRET_PARTS):
        return "[redacted]"
    if normalized_key in _RAW_PAYLOAD_KEYS and not include_raw_payloads:
        return {"omitted": True, "sha256": _hash_payload(value)}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return jsonable(value.value, include_raw_payloads=include_raw_payloads)
    if isinstance(value, BaseModel):
        return jsonable(
            value.model_dump(mode="json", by_alias=True),
            include_raw_payloads=include_raw_payloads,
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value), include_raw_payloads=include_raw_payloads)
    if isinstance(value, torch.Tensor):
        detached = value.detach().to(device="cpu")
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, Mapping):
        return {
            str(item_key): jsonable(
                item_value,
                include_raw_payloads=include_raw_payloads,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [jsonable(item, include_raw_payloads=include_raw_payloads) for item in value]
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return str(value)


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace a JSON artifact only after its complete contents reach disk."""

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
