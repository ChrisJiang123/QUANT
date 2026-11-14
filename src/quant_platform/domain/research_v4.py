"""V4 research-only cross-process envelope; not an order schema."""
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass(frozen=True)
class ResearchEnvelope:
    venue:str
    instrument_id:str
    quote_currency:str
    observed_at:datetime
    available_at:datetime
    processing_version:str
    model_version:str
    experiment_id:str
    signal_generated_at:datetime
    earliest_executable_at:datetime
    schema_version:int=1
    def __post_init__(self):
        times=[self.observed_at,self.available_at,self.signal_generated_at,self.earliest_executable_at]
        if any(t.tzinfo is None or t.utcoffset().total_seconds()!=0 for t in times):raise ValueError('explicit UTC required')
        if times!=sorted(times):raise ValueError('noncausal research envelope')
        if any(not getattr(self,k) for k in ['venue','instrument_id','quote_currency','processing_version','model_version','experiment_id']):raise ValueError('identity required')

@dataclass(frozen=True)
class TargetWeights:
    envelope:ResearchEnvelope
    weights:dict[str,float]
    @property
    def order_enabled(self):return False
    def __post_init__(self):
        if any(not math.isfinite(x) or x<0 or x>1 for x in self.weights.values()) or sum(self.weights.values())>1:raise ValueError('long/cash only')
