"""Engine-neutral run metadata and result ingestion, with sealed split identity."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ._validation import decimal, member, sha256, text, utc
from .accounting import EquityPoint, TradeFill
from .instruments import Instrument


@dataclass(frozen=True, kw_only=True)
class TimeWindow:
    """Half-open UTC interval [start, end)."""
    start_utc: datetime
    end_utc: datetime

    def __post_init__(self) -> None:
        utc(self.start_utc, "start_utc")
        utc(self.end_utc, "end_utc")
        if self.start_utc >= self.end_utc:
            raise ValueError("Empty/reversed time window")


@dataclass(frozen=True, kw_only=True)
class TestSplit:
    __test__ = False
    train: TimeWindow
    validation: TimeWindow
    holdout: TimeWindow
    holdout_seal_id: str

    def __post_init__(self) -> None:
        if not all(isinstance(w, TimeWindow) for w in (self.train, self.validation, self.holdout)):
            raise ValueError("Splits require TimeWindow objects")
        if self.train.end_utc > self.validation.start_utc or self.validation.end_utc > self.holdout.start_utc:
            raise ValueError("Splits must be chronological and nonoverlapping")
        text(self.holdout_seal_id, "holdout_seal_id")


@dataclass(frozen=True, kw_only=True)
class CostAssumptions:
    """Rates are fractions per side; None means unknown, never zero cost."""
    fee_rate: Decimal | None
    half_spread_rate: Decimal | None
    slippage_rate: Decimal | None
    fee_basis: str
    fill_policy: str
    min_notional: Decimal | None = None
    latency_seconds: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("fee_rate", "half_spread_rate", "slippage_rate", "min_notional", "latency_seconds"):
            value = getattr(self, name)
            if value is not None:
                decimal(value, name, minimum=Decimal(0))
        text(self.fee_basis, "fee_basis")
        text(self.fill_policy, "fill_policy")


@dataclass(frozen=True, kw_only=True)
class BacktestRun:
    run_id: str
    market: str
    instruments: tuple[Instrument, ...]
    strategy_version: str
    data_hashes: tuple[str, ...]
    git_commit: str | None
    engine_name: str
    engine_version: str | None
    cost_assumptions: CostAssumptions
    test_split: TestSplit
    created_at_utc: datetime

    def __post_init__(self) -> None:
        for name in ("run_id", "market", "strategy_version", "engine_name"):
            text(getattr(self, name), name)
        for name in ("git_commit", "engine_version"):
            if getattr(self, name) is not None:
                text(getattr(self, name), name)
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "data_hashes", tuple(self.data_hashes))
        if not self.instruments or not all(isinstance(i, Instrument) for i in self.instruments):
            raise ValueError("Run requires typed instruments")
        if len({i.instrument_id for i in self.instruments}) != len(self.instruments):
            raise ValueError("Duplicate run instrument")
        if not self.data_hashes:
            raise ValueError("Run requires data hashes")
        for value in self.data_hashes:
            sha256(value)
        if not isinstance(self.cost_assumptions, CostAssumptions) or not isinstance(self.test_split, TestSplit):
            raise ValueError("Run requires typed costs and test split")
        utc(self.created_at_utc, "created_at_utc")


class BacktestStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    COMPLETED = "COMPLETED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


@dataclass(frozen=True, kw_only=True)
class Measurement:
    name: str
    value: Decimal

    def __post_init__(self) -> None:
        text(self.name, "name")
        decimal(self.value, "value")


@dataclass(frozen=True, kw_only=True)
class BacktestResult:
    run: BacktestRun
    status: BacktestStatus
    completed_at_utc: datetime | None = None
    trade_ledger: tuple[TradeFill, ...] = ()
    equity_curve: tuple[EquityPoint, ...] = ()
    metrics: tuple[Measurement, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run, BacktestRun):
            raise ValueError("Result requires BacktestRun")
        member(self.status, BacktestStatus)
        for name, kind in (("trade_ledger", TradeFill), ("equity_curve", EquityPoint), ("metrics", Measurement)):
            values = tuple(getattr(self, name))
            if not all(isinstance(v, kind) for v in values):
                raise ValueError(f"Invalid {name} item")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.completed_at_utc is not None:
            utc(self.completed_at_utc, "completed_at_utc")
            if self.completed_at_utc < self.run.created_at_utc:
                raise ValueError("Result completed before run creation")
        if self.status is BacktestStatus.COMPLETED and self.completed_at_utc is None:
            raise ValueError("Completed result requires completion timestamp")
        if self.status is BacktestStatus.NOT_RUN and (self.completed_at_utc is not None or self.trade_ledger or self.equity_curve or self.metrics):
            raise ValueError("NOT_RUN cannot contain fabricated backtest outputs")
        ids = {i.instrument_id for i in self.run.instruments}
        if any(fill.instrument_id not in ids for fill in self.trade_ledger):
            raise ValueError("Fill instrument missing from run")
