"""Exercise the thin adapter against TASK001-produced synthetic local artifacts."""
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from quant_crypto.pipeline import import_file
from quant_platform.adapters.crypto.task001_data_adapter import (
    Task001DataAdapter, bar_from_task001, instrument_from_task001,
)
from quant_platform.domain import QualityStatus

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
COLLECTED = datetime(2024, 1, 5, 0, 1, tzinfo=UTC)
NORMAL = (Path(__file__).parent / "fixtures/synthetic_normal.csv").read_bytes()


def ingest(root, *, gap=False):
    source = root / "synthetic.csv"
    source.write_bytes(b"\n".join([NORMAL.splitlines()[0], NORMAL.splitlines()[2]]) if gap else NORMAL)
    manifest = import_file(source, root, symbol="XBTUSD", source_url="synthetic://NOT_MARKET_DATA",
                           collected_at=COLLECTED.isoformat(), synthetic=True)
    path = next(root / a["path"] for a in manifest["artifacts"] if a["path"].endswith(".parquet"))
    rows = pq.read_table(path).to_pylist()
    return manifest, rows


def read(adapter, instrument, decision=COLLECTED):
    return list(adapter.observations(instrument, start_utc=START, end_utc=COLLECTED, decision_time_utc=decision))


def test_adapter_retains_review_required_without_filling_or_writing(tmp_path):
    manifest, rows = ingest(tmp_path, gap=True)
    from quant_crypto.storage import file_sha256
    before = {p.relative_to(tmp_path).as_posix(): (p.stat().st_mtime_ns, file_sha256(p))
              for p in tmp_path.rglob("*") if p.is_file()}
    bars = read(Task001DataAdapter(tmp_path, manifest["dataset_id"]), instrument_from_task001(rows[0]))
    assert len(bars) == 2
    assert all(b.quality_status is QualityStatus.REVIEW_REQUIRED for b in bars)
    assert bars[1].event_time_utc - bars[0].event_time_utc == timedelta(days=2)
    assert all(b.raw_sha256 == manifest["quality"]["raw_sha256"] for b in bars)
    assert all(b.clean_version == manifest["dataset_id"] for b in bars)
    assert bars[0].source_symbol == "XXBTZUSD"
    assert "NOT_PROVEN" in bars[0].availability_basis
    assert bars[0].quote_volume is None and isinstance(bars[0].close, Decimal)
    assert "SYNTHETIC_TEST_ONLY" in bars[0].quality_flags
    after = {p.relative_to(tmp_path).as_posix(): (p.stat().st_mtime_ns, file_sha256(p))
             for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_adapter_point_in_time_and_half_open_window(tmp_path):
    manifest, rows = ingest(tmp_path)
    adapter = Task001DataAdapter(tmp_path, manifest["dataset_id"])
    instrument = instrument_from_task001(rows[0])
    available = rows[0]["available_at_utc"]
    assert read(adapter, instrument, available - timedelta(microseconds=1)) == []
    assert len(read(adapter, instrument, available)) == 1
    assert len(list(adapter.observations(instrument, start_utc=START, end_utc=START + timedelta(days=1),
                                         decision_time_utc=COLLECTED))) == 1
    # Synthetic PASS is still explicitly synthetic, not a backtest approval.
    assert manifest["quality"]["backtest_ready"] is False
    assert "SYNTHETIC_TEST_ONLY" in read(adapter, instrument)[0].quality_flags


@pytest.mark.parametrize("changes", [
    {"dataset_version": "bad"}, {"source_file_sha256": "b" * 64},
    {"is_final_candle": False}, {"quality_flags": []},
    {"venue": "ibkr"}, {"base_asset": "OTHER"},
])
def test_adapter_rejects_provenance_and_source_changes(tmp_path, changes):
    manifest, rows = ingest(tmp_path)
    with pytest.raises(ValueError):
        bar_from_task001(dict(rows[0], **changes), manifest["quality"])


def test_adapter_detects_tampered_artifact_and_unsafe_path(tmp_path):
    manifest, rows = ingest(tmp_path)
    adapter = Task001DataAdapter(tmp_path, manifest["dataset_id"])
    parquet = next(tmp_path / a["path"] for a in manifest["artifacts"] if a["path"].endswith(".parquet"))
    parquet.chmod(0o666)
    parquet.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read(adapter, instrument_from_task001(rows[0]))
    with pytest.raises(ValueError):
        adapter._local("../outside.json")


def test_adapter_rejects_changed_manifest_quality(tmp_path):
    manifest, rows = ingest(tmp_path)
    path = tmp_path / "data/manifests" / f"{manifest['dataset_id']}.json"
    manifest["quality"]["status"] = "REVIEW_REQUIRED"
    path.chmod(0o666)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="quality report mismatch"):
        read(Task001DataAdapter(tmp_path, manifest["dataset_id"]), instrument_from_task001(rows[0]))
