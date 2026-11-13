"""Local immutable research artifacts, not a remote execution protocol."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def verify(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Artifact SHA256 mismatch: {Path(path).name}")


def code_identity():
    directory = Path(__file__).parent
    files = {p.name: file_hash(p) for p in sorted(directory.glob("*.py"))}
    return {"sha256": digest(files), "files": files}


def git_commit(root):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def now():
    return datetime.now(timezone.utc).isoformat()
