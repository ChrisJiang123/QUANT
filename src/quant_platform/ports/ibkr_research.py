"""Future local-snapshot type contracts only; no adapter, SDK, credentials or connectivity."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

@dataclass(frozen=True)
class FutureOptionQuote:
    contract_id:str
    underlying_id:str
    expiration:datetime
    strike:Decimal
    put_call:str
    multiplier:Decimal
    deliverable:str
    observed_at:datetime
    available_at:datetime
    def __post_init__(self):
        if self.multiplier<=0 or self.strike<=0 or self.put_call not in {'PUT','CALL'}:raise ValueError('option identity')
        if any(x.tzinfo is None for x in [self.expiration,self.observed_at,self.available_at]) or self.available_at<self.observed_at:raise ValueError('PIT option timestamps')

class FutureEquitySnapshotPort(Protocol):
    def read_local_snapshot(self,path:str,as_of:datetime)->list[dict]:...

class FutureOptionSnapshotPort(Protocol):
    def read_local_chain(self,path:str,as_of:datetime)->list[FutureOptionQuote]:...
