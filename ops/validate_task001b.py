"""Local TASK001B verification; retain fresh evidence without overwriting TASK001.

Requires reports/task001b-before.json captured before the implementation. Never
fetches data, starts engines or connects workers/brokers. Replay exit 2 is the
existing quality-review gate, not success authorization or an import crash.
"""
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def snapshot(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def summary(path):
    suites = ET.parse(ROOT / path).getroot()
    return {key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
            for key in ("tests", "failures", "errors", "skipped")}


def main():
    baseline = json.loads((ROOT / "reports/task001b-before.json").read_text(encoding="utf-8"))
    commands = []

    def run(label, args, expected=0):
        command = [str(ROOT / ".venv/Scripts/python.exe"), *args]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = f"reports/{label}.txt"
        (ROOT / log).write_text(result.stdout + result.stderr, encoding="utf-8")
        commands.append({"command": command, "returncode": result.returncode,
                         "expected_returncode": expected, "log": log})
        print(f"{label}: exit {result.returncode} (expected {expected})")
        if result.returncode != expected:
            raise RuntimeError(f"Validation stopped: {label}; inspect {log}")

    run("pytest-task001b-original", ["-m", "pytest", "-q", "tests/test_archive.py", "tests/test_pipeline.py",
                                    "tests/test_security.py", "--junitxml=reports/pytest-task001b-original.xml"])
    run("pytest-task001b-new", ["-m", "pytest", "-q", "tests/test_multi_asset_contracts.py",
                               "tests/test_task001_data_adapter.py", "--junitxml=reports/pytest-task001b-new.xml"])
    run("pytest-task001b-after", ["-m", "pytest", "-q", "--junitxml=reports/pytest-task001b-after.xml"])
    run("pip-check-task001b", ["-m", "pip", "check"])
    receipt = "data/manifests/download-86ac87cbc20bf78b429b08e637460fbf03061947748205ce4cfca23333aed1f8.json"
    run("replay-task001b", ["ops/import_download.py", receipt], expected=2)

    # Exercise the new adapter against retained actual data, not a new acquisition.
    from quant_platform.adapters.crypto.task001_data_adapter import Task001DataAdapter
    from quant_platform.domain import AssetClass, Instrument, QualityStatus

    adapter_checks = []
    for item in json.loads((ROOT / "reports/data_catalog.json").read_text(encoding="utf-8")):
        base = {"XBTUSD": "BTC", "ETHUSD": "ETH"}[item["symbol"]]
        instrument = Instrument(instrument_id=f"crypto:kraken:spot:{base}-USD",
                                asset_class=AssetClass.CRYPTO_SPOT, venue="kraken",
                                symbol=f"{base}/USD", quote_currency="USD")
        bars = list(Task001DataAdapter(ROOT, item["dataset_id"]).observations(
            instrument, start_utc=datetime(2009, 1, 1, tzinfo=timezone.utc),
            end_utc=datetime(2026, 11, 1, tzinfo=timezone.utc),
            decision_time_utc=datetime(2026, 11, 1, tzinfo=timezone.utc)))
        passed = (len(bars) == item["rows"]
                  and all(b.quality_status is QualityStatus.REVIEW_REQUIRED
                          and b.raw_sha256 == item["raw_sha256"]
                          and b.clean_version == item["dataset_id"] for b in bars))
        adapter_checks.append({"instrument_id": instrument.instrument_id, "dataset_id": item["dataset_id"],
                               "rows": len(bars), "status": item["status"], "passed": passed})
        if not passed:
            raise RuntimeError("Actual-data adapter compatibility check failed")

    changes, missing, preserved, touched = [], [], [], []
    for relative, original in baseline["original_files"].items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        current = snapshot(path)
        if current["sha256"] != original["sha256"]:
            changes.append(relative)
        else:
            preserved.append(relative)
            if current["mtime_ns"] != original["mtime_ns"]:
                touched.append(relative)
    excluded = {".venv", "__pycache__", ".pytest_cache", ".git"}
    new_files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                       if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts)
                       and p.relative_to(ROOT).as_posix() not in baseline["original_files"])
    suites = {label: summary(f"reports/pytest-task001b-{label}.xml") for label in ("before", "original", "new", "after")}
    compatible = (baseline["returncode"] == 0 and not missing and changes == ["README.md"] and not touched
                  and suites["before"]["tests"] == suites["original"]["tests"] == 48
                  and suites["after"]["tests"] == suites["original"]["tests"] + suites["new"]["tests"]
                  and all(s["failures"] == s["errors"] == s["skipped"] == 0 for s in suites.values()))
    evidence = {"commands": commands, "pytest": suites, "actual_data_adapter": adapter_checks,
                "original_file_count": len(baseline["original_files"]),
                "preserved_files": preserved, "changed_original_files": changes,
                "missing_original_files": missing, "preserved_but_touched_files": touched,
                "new_files": new_files, "compatible": compatible,
                "replay_exit_2": "Expected REVIEW_REQUIRED; historical data gaps remain unresolved"}
    (ROOT / "reports/task001b-after.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"pytest": suites, "compatible": compatible, "original_preserved": len(preserved),
                      "original_changed": changes, "actual_data_adapter": adapter_checks}, indent=2))
    return 0 if compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
