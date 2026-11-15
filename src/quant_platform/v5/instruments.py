"""Versioned identity; incomplete venue facts never become verified margin rules."""
from dataclasses import dataclass
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import math

@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    asset: str
    multiplier: float
    tick: float
    fixed_fee: float=.0113
    quote: str='USD'
    settlement_currency: str='USD'
    linear: bool=True
    venue: str='BITNOMIAL_KRAKEN_DERIVATIVES_US'
    product_id: int|None=None
    source_available_at: str|None=None
    effective_from: str|None=None
    margin_status: str='UNVERIFIED_VENUE_SPEC'
    initial_margin_fraction: float=.20
    maintenance_fraction_of_initial: float=.50
    max_contracts: int=100000

    def validate_quantity(self,n):
        if isinstance(n,bool) or not math.isfinite(float(n)) or int(n)!=n:
            raise ValueError('integer contracts required')
        if abs(n)>self.max_contracts:raise ValueError('contract quantity limit')
        return int(n)

    def notional(self,n,price):return abs(self.validate_quantity(n))*self.multiplier*price
    def initial_margin(self,n,price):return self.notional(n,price)*self.initial_margin_fraction
    def maintenance(self,n,price):return self.initial_margin(n,price)*self.maintenance_fraction_of_initial

def cash_settlement_at(day):
    """Kraken broker cash schedule, not a guessed fixed UTC offset."""
    d=datetime.fromisoformat(str(day)[:10])
    return d.replace(hour=15,tzinfo=ZoneInfo('America/Chicago')).astimezone(timezone.utc)

def annualize_rate(rate,interval_hours):
    if not 0<interval_hours<=24: raise ValueError('actual interval length required')
    return rate*24/interval_hours*365
