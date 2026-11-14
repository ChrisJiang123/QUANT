"""Point-in-time observations with explicit source and quality provenance."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ._validation import decimal, identity, member, sha256, text, utc


class QualityStatus(str, Enum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


@dataclass(frozen=True, kw_only=True)
class MarketObservation:
    instrument_id: str
    source: str
    source_symbol: str
    event_time_utc: datetime
    available_at_utc: datetime
    collected_at_utc: datetime
    timeframe: str
    raw_sha256: str
    clean_version: str
    quality_status: QualityStatus
    availability_basis: str
    source_member_sha256: str | None = None
    quality_flags: tuple[str, ...] = ()
    calendar: str | None = None
    session: str | None = None
    adjustment_policy: str | None = None

    def __post_init__(self) -> None:
        identity(self.instrument_id)
        for name in ("source", "source_symbol", "timeframe", "clean_version", "availability_basis"):
            text(getattr(self, name), name)
        for name in ("event_time_utc", "available_at_utc", "collected_at_utc"):
            utc(getattr(self, name), name)
        if not self.event_time_utc <= self.available_at_utc <= self.collected_at_utc:
            raise ValueError("Require event <= availability <= collection")
        sha256(self.raw_sha256)
        if self.source_member_sha256 is not None:
            sha256(self.source_member_sha256)
        member(self.quality_status, QualityStatus)
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))


# Both names describe the same contract; no second schema to drift.
Observation = MarketObservation


@dataclass(frozen=True, kw_only=True)
class Bar(MarketObservation):
    close_time_utc: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    base_volume: Decimal
    quote_volume: Decimal | None
    is_final: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        utc(self.close_time_utc, "close_time_utc")
        if not self.event_time_utc < self.close_time_utc <= self.available_at_utc:
            raise ValueError("Bar must close before it is available")
        if self.is_final is not True:
            raise ValueError("Incomplete bars cannot enter research")
        for name in ("open", "high", "low", "close", "base_volume"):
            decimal(getattr(self, name), name, minimum=Decimal(0))
        if self.quote_volume is not None:
            decimal(self.quote_volume, "quote_volume", minimum=Decimal(0))
        if self.high < max(self.open, self.low, self.close) or self.low > min(self.open, self.high, self.close):
            raise ValueError("Invalid OHLC bounds")
