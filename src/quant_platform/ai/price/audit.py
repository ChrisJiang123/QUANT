"""Explicit scoped audit; never rewrite or upgrade TASK001 source verdicts."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_crypto.storage import canonical as task_canonical, sha256 as task_digest
from quant_platform.adapters.crypto.task001_data_adapter import Task001DataAdapter
from quant_platform.domain import AssetClass, Instrument

from .artifacts import digest, file_hash, verify, write_new
from .config import TrainingConfig, utc

START = utc("2019-01-01T00:00:00Z")
END = utc("2026-10-01T00:00:00Z")
SCHEMA = "audited-price-daily-v1"


def audit_bars(bars, start, end):
    """Full daily calendar audit, including endpoints, not just between-row gaps."""
    failures = []
    expected = int((end - start).total_seconds() // 86400)
    if len(bars) != expected:
        failures.append(f"Expected {expected} days, found {len(bars)}")
    for index, bar in enumerate(bars):
        if bar.event_time_utc != start + timedelta(days=index):
            failures.append(f"Missing/duplicate/out-of-order day at index {index}")
            break
        if (bar.close_time_utc != bar.event_time_utc + timedelta(days=1)
                or bar.available_at_utc != bar.close_time_utc + timedelta(seconds=60)
                or bar.collected_at_utc < bar.available_at_utc or not bar.is_final
                or min(bar.open, bar.high, bar.low, bar.close) <= 0
                or bar.base_volume < 0 or bar.high < max(bar.open, bar.low, bar.close)
                or bar.low > min(bar.open, bar.high, bar.close)
                or bar.quality_flags):
            failures.append(f"Invalid real completed OHLCV/provenance at index {index}")
    return {"status": "FAIL" if failures else "PASS", "expected_rows": expected,
            "rows": len(bars), "failures": failures, "interpolated_rows": 0}


def parent_review_disposition(quality, start):
    """Only known historical gaps entirely outside the scoped release are excluded."""
    if (quality["status"] not in ("PASS", "REVIEW_REQUIRED") or quality["synthetic_test_only"]
            or quality["invalid_rows"] or quality["duplicate_rows"] or quality["fatal_errors"]
            or quality["unfinished_rows"]):
        raise ValueError("Source has unresolved failures beyond historical excluded gaps")
    gaps = quality["gaps"]
    if sum(g["missing_candles"] for g in gaps) != quality["missing_candles"]:
        raise ValueError("Unaccounted parent missing candles")
    if quality["status"] == "REVIEW_REQUIRED" and not gaps:
        raise ValueError("Unknown REVIEW_REQUIRED reason")
    if any(utc(g["before_utc"]) > start for g in gaps):
        raise ValueError("Source gap intersects research scope")
    return {"parent_status_retained": quality["status"], "excluded_pre_scope_gaps": gaps,
            "missing_candles_excluded": quality["missing_candles"],
            "resolution": "Explicit scoped exclusion; no filling; original dataset remains unchanged"}


def create_release(root: Path):
    root = Path(root).resolve()
    config = TrainingConfig()
    catalog = json.loads((root / "reports/data_catalog.json").read_text(encoding="utf-8"))
    if {i["symbol"] for i in catalog} != {"XBTUSD", "ETHUSD"} or len(catalog) != 2:
        raise ValueError("Exactly the retained BTC/USD and ETH/USD sources are required")
    development, sealed, audits, sources = {}, {}, {}, {}
    for item in catalog:
        dataset = item["dataset_id"]
        parent_path = root / "data/manifests" / f"{dataset}.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if task_digest(task_canonical(parent["options"])) != dataset:
            raise ValueError("Source manifest identity mismatch")
        disposition = parent_review_disposition(parent["quality"], START)
        base = {"XBTUSD": "BTC", "ETHUSD": "ETH"}[item["symbol"]]
        instrument = Instrument(instrument_id=f"crypto:kraken:spot:{base}-USD", asset_class=AssetClass.CRYPTO_SPOT,
                                venue="kraken", symbol=f"{base}/USD", quote_currency="USD")
        bars = list(Task001DataAdapter(root, dataset).observations(instrument, start_utc=START,
                    end_utc=END, decision_time_utc=utc(parent["options"]["collected_at_utc"])))
        audit = audit_bars(bars, START, END)
        audits[base] = {**audit, **disposition}
        if audit["status"] != "PASS":
            raise ValueError(f"Scoped quality FAIL: {base}: {audit['failures']}")
        rows = [{"event_time_utc": b.event_time_utc.isoformat(), "available_at_utc": b.available_at_utc.isoformat(),
                 **{name: str(getattr(b, name)) for name in ("open", "high", "low", "close", "base_volume")}}
                for b in bars]
        boundary = utc(config.validation_end)
        development[base] = [r for r in rows if utc(r["event_time_utc"]) < boundary]
        # Sealed file includes past-only warmup. Validation never opens this file.
        sealed[base] = [r for r in rows if utc(r["event_time_utc"]) >= boundary - timedelta(days=91)]
        sources[base] = {"dataset_id": dataset, "manifest_sha256": file_hash(parent_path),
                         "raw_sha256": item["raw_sha256"], "parent_quality": parent["quality"]["status"]}
    recipe = {"schema": SCHEMA, "start_utc": START.isoformat(), "end_utc": END.isoformat(),
              "sources": sources, "audit": audits, "config": config.to_dict(),
              "development_sha256": digest(development), "sealed_sha256": digest(sealed)}
    release_id = digest(recipe)
    destination = root / "data/research_price" / release_id
    manifest = {"release_id": release_id, "status": "PASS", "allowed_purpose": "PRICE_PREDICTION_RESEARCH_ONLY",
                "synthetic": False, "recipe": recipe,
                "seal_id": digest({"sources": sources, "test_start": config.test_start,
                                   "test_end": config.test_end, "horizons": config.horizons}),
                "files": {"development.json": digest(development), "sealed.json": digest(sealed)},
                "limitations": ["Parent REVIEW_REQUIRED retained; all historical gaps explicitly excluded before 2019",
                                "No October 2026 data; test decisions fixed before performance inspection to available 28-day labels",
                                "Availability inferred close+60s, not proven historical publication",
                                "Decimal conversion preserves TASK001 float64 visible values only",
                                "Quality audit examines held-out data integrity, never model/test performance"]}
    if destination.exists():
        current = load_release(destination)
        if current != manifest:
            raise ValueError("Existing release identity mismatch")
        for name, expected in manifest["files"].items():
            verify(destination / name, expected)
        return destination
    destination.mkdir(parents=True)
    write_new(destination / "development.json", development)
    write_new(destination / "sealed.json", sealed)
    write_new(destination / "manifest.json", manifest)
    return destination


def load_release(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "PASS" or manifest["synthetic"]
            or manifest["allowed_purpose"] != "PRICE_PREDICTION_RESEARCH_ONLY"
            or digest(manifest["recipe"]) != manifest["release_id"]):
        raise ValueError("Formal training requires a valid PASS audited real release")
    for name, recipe_name in (("development.json", "development_sha256"), ("sealed.json", "sealed_sha256")):
        if manifest["files"][name] != manifest["recipe"][recipe_name]:
            raise ValueError("Release file hash metadata mismatch")
    if any(a["status"] != "PASS" for a in manifest["recipe"]["audit"].values()):
        raise ValueError("Scoped quality audit failed")
    return manifest
