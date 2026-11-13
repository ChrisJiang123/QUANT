"""Content addressed, append-only storage with integrity verification.

Read-only file attributes are defense in depth, not OS administrator-proof WORM.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def put_raw_file(root: Path, source: Path, expected: str) -> Path:
    """Stream whole local ZIPs, including multi-GB official full archives."""
    target = root / "data/raw" / (expected + source.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target, expected)
        return target
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    os.chmod(target, 0o444)
    verify(target, expected)
    return target


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def immutable_write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Integrity mismatch: {path.name}")
        return path
    # Exclusive creation: never overwrite an existing artifact.
    with path.open("xb") as stream:
        stream.write(data)
    os.chmod(path, 0o444)
    return path


def put_raw(root: Path, data: bytes, suffix: str) -> Path:
    if suffix not in {".csv", ".zip", ".json", ".bin", ".txt"}:
        raise ValueError("Unsupported raw extension")
    return immutable_write(root / "data/raw" / (sha256(data) + suffix), data)


def verify(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}")
