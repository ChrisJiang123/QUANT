"""Sanitized fixed-job handoff; contains public data/code and no credentials."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quant_platform.ai.price.artifacts import code_identity, file_hash, write_new
from quant_platform.ai.price.audit import load_release


def build(release, destination):
    release, destination = Path(release).resolve(), Path(destination).resolve()
    release.relative_to(ROOT)
    # Bound local creation to this project's build directory; never move/delete files.
    destination.relative_to(ROOT / "build")
    manifest = load_release(release)
    if any((release / name).exists() for name in ("development.claim.json", "final.claim.json")):
        raise ValueError("Do not export a consumed formal run or test seal")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("src", "tests"):
        shutil.copytree(ROOT/name, destination/name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "requirements-training-lock.txt"):
        shutil.copy2(ROOT/name, destination/name)
    (destination/"config").mkdir()
    for name in ("research.yaml", "price_training_v1.json"):
        shutil.copy2(ROOT/"config"/name, destination/"config"/name)
    (destination/"ops").mkdir()
    for name in ("run_price.py", "price_gpu_job.py"):
        shutil.copy2(ROOT/"ops"/name, destination/"ops"/name)
    release_relative = Path("data/research_price")/manifest["release_id"]
    target = destination/release_relative
    target.mkdir(parents=True)
    for name in ("development.json", "sealed.json", "manifest.json"):
        shutil.copy2(release/name, target/name)
    write_new(destination/"job.json", {"kind": "FIXED_PRICE_RESEARCH_JOB", "version": 1,
              "release_path": release_relative.as_posix(), "release_id": manifest["release_id"],
              "code": code_identity(), "required_gpu": "NVIDIA GeForce RTX 4090",
              "execution_authorization": "NOT_AUTHORIZED", "controller_credentials": "NOT_INCLUDED",
              "worker_mode_unchanged": "read_only_health", "automatic_next_phase": False})
    files = {p.relative_to(destination).as_posix(): file_hash(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    write_new(destination/"FILES.json", {"files": files})
    manifest_hash = file_hash(destination/"FILES.json")
    template = (ROOT/"ops/Start-PriceTraining.template.ps1").read_text(encoding="utf-8")
    launcher = destination/"Start-PriceTraining.ps1"
    launcher.write_text(template.replace("__MANIFEST_SHA256__", manifest_hash).replace("__ENV_SUFFIX__", manifest_hash[:12]), encoding="utf-8-sig")
    receipt = {"bundle": str(destination), "files": len(files), "manifest_sha256": manifest_hash,
               "launcher_sha256": file_hash(launcher), "source_code": code_identity(),
               "runtime_installer_sha256": file_hash(ROOT/"build/price-runtime/python-3.12.10-amd64.exe"),
               "runtime_delivery": "Worker downloads exact verified PSF-signed installer from python.org"}
    write_new(ROOT/f"reports/{destination.name}-bundle.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: build_price_worker.py RELEASE BUILD_DESTINATION")
    build(sys.argv[1], sys.argv[2])
