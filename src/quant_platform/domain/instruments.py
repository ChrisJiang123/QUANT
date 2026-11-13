"""Venue-specific instrument identities; fixtures do not prove real listings."""
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from ._validation import identity, member, positive, text


class AssetClass(str, Enum):
    CRYPTO_SPOT = "CRYPTO_SPOT"
    EQUITY = "EQUITY"
    ETF = "ETF"
    LISTED_OPTION = "LISTED_OPTION"


class PutCall(str, Enum):
    PUT = "PUT"
    CALL = "CALL"


@dataclass(frozen=True, kw_only=True)
class OptionSpec:
    expiration: date
    strike: Decimal
    put_call: PutCall
    contract_multiplier: Decimal
    exercise_style: str | None = None
    settlement_type: str | None = None
    deliverable: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expiration, date) or isinstance(self.expiration, datetime):
            raise ValueError("expiration must be a contract date")
        positive(self.strike, "strike")
        positive(self.contract_multiplier, "contract_multiplier")
        member(self.put_call, PutCall)
        for name in ("exercise_style", "settlement_type", "deliverable"):
            value = getattr(self, name)
            if value is not None:
                text(value, name)


@dataclass(frozen=True, kw_only=True)
class Instrument:
    instrument_id: str
    asset_class: AssetClass
    venue: str
    symbol: str
    quote_currency: str
    price_increment: Decimal | None = None
    quantity_increment: Decimal | None = None
    underlying_id: str | None = None
    option_spec: OptionSpec | None = None

    def __post_init__(self) -> None:
        member(self.asset_class, AssetClass)
        _, venue, asset_type, _ = identity(self.instrument_id)
        expected = {AssetClass.CRYPTO_SPOT: "spot", AssetClass.EQUITY: "stock",
                    AssetClass.ETF: "etf", AssetClass.LISTED_OPTION: "option"}
        if venue != self.venue or asset_type != expected[self.asset_class]:
            raise ValueError("instrument_id venue/asset_type disagree with metadata")
        for name in ("venue", "symbol", "quote_currency"):
            text(getattr(self, name), name)
        for name in ("price_increment", "quantity_increment"):
            value = getattr(self, name)
            if value is not None:
                positive(value, name)
        if self.asset_class is AssetClass.LISTED_OPTION:
            if not isinstance(self.option_spec, OptionSpec) or self.underlying_id is None:
                raise ValueError("Listed options require option_spec and underlying_id")
            identity(self.underlying_id)
            if self.underlying_id == self.instrument_id:
                raise ValueError("Option cannot be its own underlying")
        elif self.option_spec is not None or self.underlying_id is not None:
            raise ValueError("Option fields belong only to listed options")
