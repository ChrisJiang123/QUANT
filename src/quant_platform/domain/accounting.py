"""Normalized *simulated* fills and equity values, never order requests."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ._validation import decimal, identity, member, positive, text, utc


class FillSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, kw_only=True)
class TradeFill:
    instrument_id: str
    time_utc: datetime
    side: FillSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str

    def __post_init__(self) -> None:
        identity(self.instrument_id)
        utc(self.time_utc, "time_utc")
        member(self.side, FillSide)
        positive(self.quantity, "quantity")
        decimal(self.price, "price", minimum=Decimal(0))
        # A measured fee may be negative for a rebate; do not erase that fact.
        decimal(self.fee, "fee")
        text(self.fee_currency, "fee_currency")


@dataclass(frozen=True, kw_only=True)
class EquityPoint:
    time_utc: datetime
    equity: Decimal
    currency: str

    def __post_init__(self) -> None:
        utc(self.time_utc, "time_utc")
        decimal(self.equity, "equity")
        text(self.currency, "currency")
