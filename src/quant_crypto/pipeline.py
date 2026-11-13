"""Offline, deterministic Kraken 1440-minute CSV/ZIP validation and normalization."""
import csv
import hashlib
import io
import json
import math
import platform
from importlib.metadata import version
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .storage import canonical, file_sha256, immutable_write, put_raw, put_raw_file, sha256, verify

UTC = timezone.utc
DAY = timedelta(days=1)
SCHEMA_VERSION = "kraken-1d-v1"
SYMBOLS = {"XBTUSD": ("BTC", "XXBTZUSD"), "ETHUSD": ("ETH", "XETHZUSD")}
SCHEMA = pa.schema([
    *[(n, pa.string()) for n in ("venue", "market_type", "venue_symbol", "base_asset", "quote_asset", "timeframe")],
    *[(n, pa.timestamp("us", tz="UTC")) for n in ("event_open_ts_utc", "event_close_ts_utc", "available_at_utc", "collected_at_utc")],
    *[(n, pa.float64()) for n in ("open", "high", "low", "close", "base_volume", "quote_volume")],
    ("trade_count", pa.int64()),
    *[(n, pa.string()) for n in ("source_url", "source_file_sha256", "source_member_sha256", "batch_id", "dataset_version", "ingestion_code_git_sha", "availability_basis")],
    ("is_final_candle", pa.bool_()), ("quality_flags", pa.list_(pa.string())),
])


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone offset required")
    return result.astimezone(UTC)


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha(root):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    except OSError:
        return "UNVERSIONED_GIT_UNAVAILABLE"
    return result.stdout.strip() if result.returncode == 0 else "UNVERSIONED_NO_GIT_REPOSITORY"


def number(value):
    result = float(Decimal(value))
    if not math.isfinite(result):
        raise ValueError("Nonfinite numeric field")
    return result


def parse_row(fields, collected):
    if len(fields) != 7:
        raise ValueError("Expected headerless Kraken timestamp,open,high,low,close,volume,trades")
    epoch = Decimal(fields[0])
    if not epoch.is_finite() or epoch != epoch.to_integral_value():
        raise ValueError("Epoch must be integer UNIX seconds")
    timestamp = datetime.fromtimestamp(int(epoch), UTC)
    if timestamp < datetime(2009, 1, 1, tzinfo=UTC) or timestamp > collected + DAY:
        raise ValueError("Impossible timestamp or wrong epoch unit")
    if int(epoch) % 86400:
        raise ValueError("Daily candle must align to UTC midnight")
    o, h, l, c, v = map(number, fields[1:6])
    count = Decimal(fields[6])
    if not count.is_finite() or count != count.to_integral_value() or not 0 <= count <= 2**63 - 1:
        raise ValueError("Invalid trade count")
    if min(o, h, l, c) <= 0 or v < 0 or h < max(o, l, c) or l > min(o, h, c):
        raise ValueError("Invalid OHLCV prices, bounds or volume")
    return timestamp, o, h, l, c, v, int(count)


def import_file(path: Path, root: Path, *, symbol: str, source_url: str,
                collected_at: str, expected_sha256: str | None = None,
                member: str | None = None, as_of: str | None = None,
                publication_lag_seconds: int = 60, provenance: str | None = None,
                synthetic: bool = False):
    root = root.resolve()
    if symbol not in SYMBOLS:
        raise ValueError("Only XBTUSD and ETHUSD spot daily research allowed")
    if publication_lag_seconds < 0 or publication_lag_seconds > 86400:
        raise ValueError("Invalid publication lag")
    collected = utc(collected_at)
    cutoff = utc(as_of) if as_of else collected
    if cutoff > collected:
        raise ValueError("as_of must not exceed collection time")
    raw_hash = file_sha256(path)
    if expected_sha256 and raw_hash != expected_sha256:
        raise ValueError("SHA256 mismatch before import")
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".zip"}:
        raise ValueError("Only CSV/ZIP supported")
    raw_path = put_raw_file(root, path, raw_hash)
    options = {"source_sha256": raw_hash, "symbol": symbol, "source_url": source_url,
               "collected_at_utc": collected.isoformat(), "as_of_utc": cutoff.isoformat(),
               "publication_lag_seconds": publication_lag_seconds, "member": member,
               "provenance": provenance, "synthetic": synthetic, "schema_version": SCHEMA_VERSION,
               "code_sha256": code_fingerprint(), "python": platform.python_version(),
               "pyarrow": pa.__version__, "tzdata": version("tzdata"), "package_version": __version__, "git_sha": git_sha(root)}
    dataset_id = sha256(canonical(options))
    manifest_path = root / "data/manifests" / f"{dataset_id}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["options"] != options:
            raise ValueError("Manifest identity mismatch")
        for artifact in manifest["artifacts"]:
            verify(root / artifact["path"], artifact["sha256"])
        return manifest
    quarantine, parsed, fatal = [], [], []
    csv_data, member_hash = b"", raw_hash
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(raw_path) as archive:
                target = member or f"{symbol}_1440.csv"
                matches = [n for n in archive.namelist() if n == target]
                if len(matches) != 1 or ".." in Path(target).parts or Path(target).is_absolute():
                    raise ValueError("ZIP must contain one exact safe requested member")
                info = archive.getinfo(target)
                if info.file_size > 16 * 1024 * 1024 or info.flag_bits & 1:
                    raise ValueError("Oversized/encrypted ZIP member forbidden")
                csv_data = archive.read(target)
                member_hash = sha256(csv_data)
                put_raw(root, csv_data, ".csv")
        else:
            if raw_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Daily CSV exceeds safety bound")
            csv_data = raw_path.read_bytes()
        rows = list(csv.reader(io.StringIO(csv_data.decode("utf-8-sig"))))
        previous = None
        for line, fields in enumerate(rows, 1):
            try:
                values = parse_row(fields, collected)
                if previous is not None and values[0] < previous:
                    fatal.append(f"Unsorted timestamp at line {line}")
                previous = values[0]
                parsed.append((line, fields, values))
            except (ValueError, InvalidOperation, OverflowError, OSError) as exc:
                quarantine.append({"line": line, "fields": fields, "reason": "INVALID_ROW", "detail": str(exc)})
    except (ValueError, UnicodeError, zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError, OSError) as exc:
        fatal.append(f"Invalid archive/CSV: {exc}")
    counts = {}
    for _, _, values in parsed:
        counts[values[0]] = counts.get(values[0], 0) + 1
    accepted, valid_times = [], []
    duplicate_rows, unfinished = 0, 0
    for line, fields, values in parsed:
        timestamp, o, h, l, c, v, trades = values
        close_time = timestamp + DAY
        available = close_time + timedelta(seconds=publication_lag_seconds)
        if counts[timestamp] > 1:
            duplicate_rows += 1
            quarantine.append({"line": line, "fields": fields, "reason": "DUPLICATE_KEY"})
            continue
        if available > cutoff:
            unfinished += 1
            quarantine.append({"line": line, "fields": fields, "reason": "UNFINISHED_OR_NOT_YET_AVAILABLE"})
            continue
        valid_times.append(timestamp)
        accepted.append(dict(zip(SCHEMA.names, [
            "kraken", "spot", SYMBOLS[symbol][1], SYMBOLS[symbol][0], "USD", "1d",
            timestamp, close_time, available, collected, o, h, l, c, v, None, trades,
            source_url, raw_hash, member_hash, dataset_id, dataset_id, options["git_sha"],
            "INFERRED_BAR_CLOSE_PLUS_LAG_NOT_PROVEN_HISTORICAL_AVAILABILITY", True,
            ["SYNTHETIC_TEST_ONLY"] if synthetic else [],
        ])))
    accepted.sort(key=lambda x: x["event_open_ts_utc"])
    times = sorted(valid_times)
    gaps = [{"after_utc": a.isoformat(), "before_utc": b.isoformat(), "missing_candles": (b - a).days - 1}
            for a, b in zip(times, times[1:]) if b - a > DAY]
    invalid = sum(q["reason"] == "INVALID_ROW" for q in quarantine)
    status = "FAIL" if fatal or invalid or duplicate_rows or not accepted else "REVIEW_REQUIRED" if gaps else "PASS"
    report = {"dataset_id": dataset_id, "symbol": symbol, "synthetic_test_only": synthetic,
              "status": status, "backtest_ready": status == "PASS" and not synthetic,
              "input_rows": len(parsed) + invalid, "accepted_rows": len(accepted),
              "invalid_rows": invalid, "duplicate_rows": duplicate_rows, "unfinished_rows": unfinished,
              "missing_candles": sum(g["missing_candles"] for g in gaps), "gaps": gaps, "fatal_errors": fatal,
              "first_event_open_utc": times[0].isoformat() if times else None,
              "last_event_open_utc": times[-1].isoformat() if times else None,
              "available_at_rule": f"bar_close + {publication_lag_seconds}s, inferred, not proven; intrabar/PIT availability analysis forbidden",
              "quote_volume": "null: archive provides base volume, no measured quote turnover",
              "gap_policy": "No fill; gaps may be no-trade intervals; review before backtesting",
              "raw_sha256": raw_hash, "member_sha256": member_hash}
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(accepted, schema=SCHEMA), sink, compression="zstd")
    outputs = {
        f"data/normalized/{dataset_id}/{symbol}-1d.parquet": sink.getvalue().to_pybytes(),
        f"data/quarantine/{dataset_id}.json": canonical(quarantine),
        f"reports/{dataset_id}.json": canonical(report),
        f"reports/{dataset_id}.md": (f"# Data quality — {symbol}\n\nStatus: **{status}**; synthetic test only: **{synthetic}**\n\n"
            f"Accepted: {len(accepted)}; invalid: {invalid}; duplicate: {duplicate_rows}; unfinished: {unfinished}; missing: {report['missing_candles']}.\n\n"
            f"Coverage (UTC candle opens): {report['first_event_open_utc']} → {report['last_event_open_utc']}.\n\n"
            f"Availability: {report['available_at_rule']}.\n\nQuote volume is null. No gap filling. See matching JSON for all gap intervals/errors.\n").encode(),
    }
    artifacts = [{"path": raw_path.relative_to(root).as_posix(), "sha256": raw_hash}]
    if provenance:
        provenance_path = (root / provenance).resolve()
        provenance_path.relative_to(root)
        artifacts.append({"path": provenance, "sha256": sha256(provenance_path.read_bytes())})
    if suffix == ".zip":
        artifacts.append({"path": f"data/raw/{member_hash}.csv", "sha256": member_hash}) if not fatal else None
    for relative, content in outputs.items():
        immutable_write(root / relative, content)
        artifacts.append({"path": relative, "sha256": sha256(content)})
    manifest = {"dataset_id": dataset_id, "options": options, "quality": report, "artifacts": artifacts}
    immutable_write(manifest_path, canonical(manifest))
    return manifest
