import io
import json
import os
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from quant_crypto.pipeline import import_file
from quant_crypto.storage import sha256

NORMAL = (Path(__file__).parent / "fixtures/synthetic_normal.csv").read_bytes()


def ingest(tmp_path, data=NORMAL, suffix=".csv", **overrides):
    source = tmp_path / ("synthetic" + suffix)
    source.write_bytes(data)
    options = dict(symbol="XBTUSD", source_url="synthetic://fixture/NOT_MARKET_DATA",
                   collected_at="2024-01-05T00:01:00Z", synthetic=True)
    options.update(overrides)
    return import_file(source, tmp_path, **options)


def test_normal_utc_schema_and_replay(tmp_path):
    first = ingest(tmp_path)
    assert first["quality"]["status"] == "PASS"
    assert first["quality"]["accepted_rows"] == 3
    assert not first["quality"]["backtest_ready"]
    table = pq.read_table(tmp_path / first["artifacts"][1]["path"])
    assert str(table.schema.field("event_open_ts_utc").type) == "timestamp[us, tz=UTC]"
    row = table.to_pylist()[0]
    assert row["base_asset"] == "BTC" and row["venue_symbol"] == "XXBTZUSD"
    assert row["quote_volume"] is None
    assert row["quality_flags"] == ["SYNTHETIC_TEST_ONLY"]
    assert row["available_at_utc"].isoformat() == "2024-01-02T00:01:00+00:00"
    before = {str(p): (p.stat().st_mtime_ns, sha256(p.read_bytes())) for p in tmp_path.rglob("*") if p.is_file() and p.name != "synthetic.csv"}
    assert ingest(tmp_path) == first
    after = {str(p): (p.stat().st_mtime_ns, sha256(p.read_bytes())) for p in tmp_path.rglob("*") if p.is_file() and p.name != "synthetic.csv"}
    assert before == after


def test_gap_no_fill(tmp_path):
    manifest = ingest(tmp_path, b"\n".join([NORMAL.splitlines()[0], NORMAL.splitlines()[2]]))
    assert manifest["quality"]["status"] == "REVIEW_REQUIRED"
    assert manifest["quality"]["missing_candles"] == 1
    assert manifest["quality"]["accepted_rows"] == 2


@pytest.mark.parametrize("timestamp", ["1704067200000", "-1000000", "1704067201", "NaN", "999999999999999999999999", "1704067200.5"])
def test_timestamp_errors(tmp_path, timestamp):
    m = ingest(tmp_path, f"{timestamp},100,110,90,105,10,3\n".encode())
    assert m["quality"]["status"] == "FAIL"
    assert m["quality"]["invalid_rows"] == 1


@pytest.mark.parametrize("fields", ["0,110,90,105,10,3", "100,99,90,105,10,3", "100,110,106,105,10,3", "100,110,90,105,-1,3", "NaN,110,90,105,10,3", "100,inf,90,105,10,3", "100,110,90,105,10,1.5", "100,110,90,105,10,-1"])
def test_abnormal_prices_and_counts(tmp_path, fields):
    m = ingest(tmp_path, f"1704067200,{fields}\n".encode())
    assert m["quality"]["status"] == "FAIL"
    assert m["quality"]["accepted_rows"] == 0


def test_duplicate_all_copies_quarantined(tmp_path):
    m = ingest(tmp_path, NORMAL + NORMAL.splitlines()[0] + b"\n")
    assert m["quality"]["duplicate_rows"] == 2
    assert m["quality"]["accepted_rows"] == 2
    assert m["quality"]["status"] == "FAIL"


def test_unfinished_candle_quarantined(tmp_path):
    m = ingest(tmp_path, collected_at="2024-01-03T12:00:00Z")
    assert m["quality"]["unfinished_rows"] == 1
    assert m["quality"]["accepted_rows"] == 2


def test_close_without_publication_lag_not_available(tmp_path):
    m = ingest(tmp_path, collected_at="2024-01-04T00:00:00Z")
    assert m["quality"]["unfinished_rows"] == 1


def test_hash_mismatch_before_ingest(tmp_path):
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ingest(tmp_path, expected_sha256="0" * 64)
    assert not (tmp_path / "data/raw").exists()


def test_corrupt_zip(tmp_path):
    m = ingest(tmp_path, b"PK\x03\x04broken", ".zip")
    assert m["quality"]["status"] == "FAIL"
    assert m["quality"]["fatal_errors"]


def make_zip(name, content=NORMAL):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, content)
    return out.getvalue()


def test_zip_import_and_parent_member_hash(tmp_path):
    m = ingest(tmp_path, make_zip("ETHUSD_1440.csv"), ".zip", symbol="ETHUSD")
    assert m["quality"]["status"] == "PASS"
    assert m["quality"]["member_sha256"] == sha256(NORMAL)
    assert m["quality"]["raw_sha256"] != sha256(NORMAL)


def test_zip_crc_corruption(tmp_path):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("XBTUSD_1440.csv", NORMAL)
    damaged = out.getvalue().replace(b"100,110", b"101,110", 1)
    m = ingest(tmp_path, damaged, ".zip")
    assert m["quality"]["status"] == "FAIL"
    assert "CRC" in str(m["quality"]["fatal_errors"])


def test_zip_traversal_not_extracted(tmp_path):
    m = ingest(tmp_path, make_zip("../escape.csv"), ".zip", member="../escape.csv")
    assert m["quality"]["status"] == "FAIL"
    assert not (tmp_path.parent / "escape.csv").exists()


def test_unsorted_is_fatal(tmp_path):
    m = ingest(tmp_path, b"\n".join(reversed(NORMAL.splitlines())))
    assert m["quality"]["status"] == "FAIL"


def test_output_tampering_detected_on_replay(tmp_path):
    m = ingest(tmp_path)
    artifact = tmp_path / m["artifacts"][1]["path"]
    os.chmod(artifact, 0o666)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ingest(tmp_path)


def test_raw_tampering_detected(tmp_path):
    m = ingest(tmp_path)
    raw = tmp_path / m["artifacts"][0]["path"]
    os.chmod(raw, 0o666)
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ingest(tmp_path)


def test_timezone_required(tmp_path):
    with pytest.raises(ValueError, match="Timezone"):
        ingest(tmp_path, collected_at="2024-01-05T00:00:00")


def test_offset_is_normalized(tmp_path):
    m = ingest(tmp_path, collected_at="2024-01-04T16:01:00-08:00")
    assert m["options"]["collected_at_utc"] == "2024-01-05T00:01:00+00:00"


def test_unapproved_symbol(tmp_path):
    with pytest.raises(ValueError, match="Only XBTUSD"):
        ingest(tmp_path, symbol="SOLUSD")


def test_empty_input_fails(tmp_path):
    assert ingest(tmp_path, b"")["quality"]["status"] == "FAIL"
