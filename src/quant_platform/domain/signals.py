"""Research intent and simulation timing; never a broker instruction."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ._validation import decimal, identity, member, text, utc


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True, kw_only=True)
class ResearchSignal:
    strategy_id: str
    instrument_id: str
    decision_time_utc: datetime
    available_at_utc: datetime
    earliest_action_time_utc: datetime
    direction: Direction
    target_exposure: Decimal
    data_version: str
    model_version: str | None
    reason: str
    confidence: Decimal | None = None

    def __post_init__(self) -> None:
        identity(self.instrument_id)
        for name in ("strategy_id", "data_version", "reason"):
            text(getattr(self, name), name)
        if self.model_version is not None:
            text(self.model_version, "model_version")
        for name in ("decision_time_utc", "available_at_utc", "earliest_action_time_utc"):
            utc(getattr(self, name), name)
        if not self.available_at_utc <= self.decision_time_utc <= self.earliest_action_time_utc:
            raise ValueError("Require available_at <= decision <= earliest_action")
        member(self.direction, Direction)
        decimal(self.target_exposure, "target_exposure")
        if self.confidence is not None:
            decimal(self.confidence, "confidence", minimum=Decimal(0))
            if self.confidence > 1:
                raise ValueError("confidence must be <= 1")

    def validate_action_time(self, simulated_time_utc: datetime) -> None:
        """Validate a simulated action timestamp; this method performs no action."""
        utc(simulated_time_utc, "simulated_time_utc")
        if simulated_time_utc < self.earliest_action_time_utc:
            raise ValueError("Simulated action precedes earliest_action_time_utc")
