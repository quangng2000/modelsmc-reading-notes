"""Create a deterministic research archive with SHA256SUMS."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(source: Path, output: Path) -> list[Path]:
    result: list[Path] = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing to archive symlink: {path}")
        if path.is_file() and path.resolve() != output.resolve():
            result.append(path)
    return sorted(result, key=lambda path: path.relative_to(source).as_posix())


def create_bundle(source: Path, output: Path) -> dict[str, object]:
    """Bundle a directory with normalized metadata and checksum inventory."""

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"bundle source is not a directory: {source}")
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _files(source, output)
    checksums = [f"{_sha256(path)}  {path.relative_to(source).as_posix()}" for path in files]
    sums_content = ("\n".join(checksums) + "\n").encode()
    with output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(source).as_posix()
                    info = archive.gettarinfo(str(path), arcname=f"{source.name}/{relative}")
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                sums_info = tarfile.TarInfo(f"{source.name}/SHA256SUMS")
                sums_info.size = len(sums_content)
                sums_info.mode = 0o644
                sums_info.mtime = 0
                archive.addfile(sums_info, io.BytesIO(sums_content))
    archive_sha256 = _sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8")
    return {
        "source": str(source),
        "archive": str(output),
        "files": len(files),
        "archive_sha256": archive_sha256,
        "checksum_sidecar": str(sidecar),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = create_bundle(args.source, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
