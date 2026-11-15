import importlib.util
import pandas as pd
from quant_platform.v5.common import ROOT,OUT,read
from quant_platform.research.governance import digest_file

spec=importlib.util.spec_from_file_location('v5_public_collection',ROOT/'scripts/v5_collect_us.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_unclosed_or_incomplete_daily_bar_cannot_enter_public_feature_store():
    s=dict(pair='X',instrument_id='TEST:X',product_id=1)
    raw=[dict(date='2024-01-01',ohlc=dict(open=1,high=2,low=1,close=2)),dict(date='2024-01-02',ohlc=dict(open=2,high=3,low=2,close=3)),dict(date='2023-12-31',ohlc=dict(open=1,high=2,low=1))]
    rows=module.closed_prices(raw,s,'2024-01-02T10:00:00Z','PUBLIC_FIXTURE')
    assert len(rows)==1
    assert rows[0]['event_at']<rows[0]['observed_at']<=rows[0]['available_at']==rows[0]['received_at']
    assert rows[0]['observed_at']==pd.Timestamp('2024-01-02',tz='UTC')

def test_real_append_capture_preserves_clock_hashes_and_never_claims_forward():
    pointer=read(OUT/'us_collection_latest.json');p=ROOT/pointer['result'];assert digest_file(p)==pointer['hash']
    data=read(p);assert data['financial_evidence'] is False
    assert data['us_status']=='BLOCKED_US_FORWARD'
    assert p.parent.parent==OUT/'us_collections'
    for name,h in data['files'].items():assert digest_file(p.parent/name)==h
    for name in ['XBTUSD_closed_prices.parquet','ETHUSD_closed_prices.parquet']:
        f=pd.read_parquet(p.parent/name)
        assert (f.observed_at<=f.available_at).all() and (f.received_at==f.available_at).all()
        assert (f.available_at>=pd.Timestamp(data['captured_at'])).all()
