"""Read existing immutable TASK001 artifacts; do not ingest, fill, or rewrite data."""
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping, Any

from quant_platform.domain import AssetClass, Bar, Instrument, QualityStatus, TimeWindow
from quant_platform.domain._validation import sha256, utc


def instrument_from_task001(row: Mapping[str, Any]) -> Instrument:
    # Reuse TASK001's source identity registry instead of a second market pipeline.
    from quant_crypto.pipeline import SYMBOLS

    known = {(base, venue_symbol) for base, venue_symbol in SYMBOLS.values()}
    if (row["venue"] != "kraken" or row["market_type"] != "spot"
            or row["quote_asset"] != "USD" or row["timeframe"] != "1d"
            or (row["base_asset"], row["venue_symbol"]) not in known):
        raise ValueError("Unsupported TASK001 source identity")
    base = row["base_asset"]
    return Instrument(instrument_id=f"crypto:kraken:spot:{base}-USD",
                      asset_class=AssetClass.CRYPTO_SPOT, venue="kraken",
                      symbol=f"{base}/USD", quote_currency="USD")


def bar_from_task001(row: Mapping[str, Any], quality: Mapping[str, Any]) -> Bar:
    """Translate one accepted row and its *dataset-level* report; never infer PASS."""
    instrument = instrument_from_task001(row)
    if (row["dataset_version"] != quality["dataset_id"]
            or row["batch_id"] != quality["dataset_id"]
            or row["source_file_sha256"] != quality["raw_sha256"]
            or row["source_member_sha256"] != quality["member_sha256"]):
        raise ValueError("TASK001 row/report provenance mismatch")
    flags = tuple(row["quality_flags"])
    if quality["synthetic_test_only"] and "SYNTHETIC_TEST_ONLY" not in flags:
        raise ValueError("Synthetic provenance flag lost")
    # TASK001 stores float64; Decimal(str(...)) preserves its visible value,
    # not precision already lost upstream. Raw bytes remain the numeric authority.
    values = {name: Decimal(str(row[name])) for name in ("open", "high", "low", "close", "base_volume")}
    return Bar(
        instrument_id=instrument.instrument_id, source=row["source_url"],
        source_symbol=row["venue_symbol"], event_time_utc=row["event_open_ts_utc"],
        close_time_utc=row["event_close_ts_utc"], available_at_utc=row["available_at_utc"],
        collected_at_utc=row["collected_at_utc"], timeframe=row["timeframe"],
        raw_sha256=row["source_file_sha256"], source_member_sha256=row["source_member_sha256"],
        clean_version=row["dataset_version"], quality_status=QualityStatus(quality["status"]),
        availability_basis=row["availability_basis"], quality_flags=flags,
        calendar="UTC_24X7", session="24X7", adjustment_policy="UNADJUSTED_SPOT",
        quote_volume=None if row["quote_volume"] is None else Decimal(str(row["quote_volume"])),
        is_final=row["is_final_candle"], **values,
    )


class Task001DataAdapter:
    """Local read-only MarketDataPort. REVIEW_REQUIRED remains inspectable, not approved."""

    def __init__(self, root: Path, dataset_id: str):
        sha256(dataset_id)
        self.root = Path(root).resolve()
        self.dataset_id = dataset_id

    def _local(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        path.relative_to(self.root)
        return path

    def observations(
        self, instrument: Instrument, *, start_utc: datetime, end_utc: datetime,
        decision_time_utc: datetime,
    ) -> Iterable[Bar]:
        # Dependencies are already installed by TASK001, loaded only on explicit use.
        import pyarrow.parquet as pq
        from quant_crypto.pipeline import SCHEMA
        from quant_crypto.storage import canonical, sha256 as digest, verify

        window = TimeWindow(start_utc=start_utc, end_utc=end_utc)
        utc(decision_time_utc, "decision_time_utc")
        manifest = json.loads(self._local(f"data/manifests/{self.dataset_id}.json").read_text(encoding="utf-8"))
        if manifest["dataset_id"] != self.dataset_id or digest(canonical(manifest["options"])) != self.dataset_id:
            raise ValueError("TASK001 manifest identity mismatch")
        artifacts = {a["path"]: a["sha256"] for a in manifest["artifacts"]}
        for relative, expected in artifacts.items():
            verify(self._local(relative), expected)
        quality = manifest["quality"]
        report_path = f"reports/{self.dataset_id}.json"
        if report_path not in artifacts or json.loads(self._local(report_path).read_text(encoding="utf-8")) != quality:
            raise ValueError("TASK001 quality report mismatch")
        symbol = manifest["options"]["symbol"]
        relative = f"data/normalized/{self.dataset_id}/{symbol}-1d.parquet"
        if relative not in artifacts:
            raise ValueError("Missing normalized artifact hash")
        table = pq.read_table(self._local(relative))
        if not table.schema.equals(SCHEMA):
            raise ValueError("Unsupported TASK001 Parquet schema")
        for row in table.to_pylist():
            if instrument_from_task001(row) != instrument:
                raise ValueError("Requested instrument differs from dataset instrument")
            bar = bar_from_task001(row, quality)
            if window.start_utc <= bar.event_time_utc < window.end_utc and bar.available_at_utc <= decision_time_utc:
                yield bar
