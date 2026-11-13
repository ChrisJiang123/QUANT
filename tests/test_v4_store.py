from datetime import datetime,timezone,timedelta
import pandas as pd
import pytest
from quant_platform.features.store import FeatureStore
from quant_platform.domain.research_v4 import ResearchEnvelope,TargetWeights

def test_feature_store_rejects_future_and_preserves_version(tmp_path):
    t=pd.Timestamp('2020-01-01',tz='UTC')
    rows=pd.DataFrame(dict(pair=['XBTUSD'],factor=['mom7'],value=[.1],observed_at=[t],available_at=[t+pd.Timedelta(days=1)],processing_version=['v4']))
    store=FeatureStore(tmp_path);path=store.append(rows)
    assert store.query('XBTUSD','mom7',t).empty
    assert store.query('XBTUSD','mom7',t+pd.Timedelta(days=1)).iloc[-1].processing_version=='v4'
    assert store.append(rows)==path

def test_research_envelope_and_weights_are_never_orders():
    t=datetime(2020,1,1,tzinfo=timezone.utc)
    r=ResearchEnvelope(venue='kraken',instrument_id='crypto:kraken:spot:BTC-USD',quote_currency='USD',
                       observed_at=t,available_at=t,processing_version='v4',model_version='m',experiment_id='e',
                       signal_generated_at=t,earliest_executable_at=t+timedelta(days=1))
    w=TargetWeights(envelope=r,weights={'BTC/USD':.4,'ETH/USD':.4});assert w.order_enabled is False
    with pytest.raises(ValueError):TargetWeights(envelope=r,weights={'BTC/USD':1.1})
