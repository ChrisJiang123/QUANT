"""Execute tests, offline actual-data replay and retain command exits/evidence."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from quant_crypto.pipeline import code_fingerprint
from quant_crypto.storage import file_sha256

root = Path(__file__).resolve().parents[1]
receipt = root / "data/manifests/download-86ac87cbc20bf78b429b08e637460fbf03061947748205ce4cfca23333aed1f8.json"
runs = []


def run(label, args):
    result = subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (root / f"reports/{label}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    runs.append({"label": label, "command": [".venv/Scripts/python.exe", *args], "exit_code": result.returncode, "log": f"reports/{label}.txt"})
    print(label, "exit", result.returncode)
    return result


tests = run("pytest-task001", ["-m", "pytest", "-q", "--junitxml=reports/pytest-task001.xml"])
first = run("import-task001", ["ops/import_download.py", str(receipt.relative_to(root))])
files = [p for d in ("data/raw", "data/normalized", "data/manifests", "data/quarantine") for p in (root / d).rglob("*") if p.is_file()]
before = {p.relative_to(root).as_posix(): [p.stat().st_mtime_ns, file_sha256(p)] for p in files}
second = run("replay-task001", ["ops/import_download.py", str(receipt.relative_to(root))])
files_after = [p for d in ("data/raw", "data/normalized", "data/manifests", "data/quarantine") for p in (root / d).rglob("*") if p.is_file()]
after = {p.relative_to(root).as_posix(): [p.stat().st_mtime_ns, file_sha256(p)] for p in files_after}
dependencies = run("pip-check-task001", ["-m", "pip", "check"])
suites = ET.parse(root / "reports/pytest-task001.xml").getroot()
summary = {name: sum(int(s.get(name, 0)) for s in suites.iter("testsuite")) for name in ("tests", "failures", "errors", "skipped")}
quality = []
for p in (root / "data/manifests").glob("*.json"):
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("options", {}).get("code_sha256") == code_fingerprint():
        quality.append(data["quality"])
evidence = {"code_sha256": code_fingerprint(), "commands": runs, "pytest": summary,
            "actual_data_idempotent": before == after and first.stdout == second.stdout,
            "data_files_checked": len(before), "quality": quality,
            "import_exit_2_meaning": "Quality review gate: historical gaps retained; not a crashed import"}
(root / "reports/execution_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps({"pytest": summary, "actual_data_idempotent": evidence["actual_data_idempotent"]}, indent=2))
raise SystemExit(0 if tests.returncode == dependencies.returncode == 0 and first.returncode == second.returncode == 2 and evidence["actual_data_idempotent"] and len(quality) == 2 else 1)
