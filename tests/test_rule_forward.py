import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quant_platform.research.governance import Ledger, offline_guard
from quant_platform.research.rule_forward import (DAY, append_unique, historical_signals,
    independent_windows, latest_rule, make_observation, next_open, settle, validate_snapshot)

ROOT=Path(__file__).resolve().parents[1]

def bars(n=400):
    times=pd.date_range('2020-01-01',periods=n,tz='UTC')
    frames=[]
    for j,p in enumerate(['ETHUSD','XBTUSD']):
        close=(100+np.arange(n)*.2+np.sin(np.arange(n)))*(j+1)
        frames.append(pd.DataFrame(dict(pair=p,candle_ts=times,open=close-.1,
                     high=close+2,low=close-2,close=close,volume=100000.,
                     available_at=times+DAY)))
    return pd.concat(frames,ignore_index=True)

def prediction(tmp_path):
    b=bars(); at=b.candle_ts.max()+DAY+pd.Timedelta(hours=2)
    b['available_at']=at-pd.Timedelta(minutes=1)
    p=make_observation(b,at,'fixture','isolated-fixture','fixture-hash')
    r,_=append_unique(tmp_path/'p.jsonl','fixture-day',p,p['dependency_hash'])
    return r,b,at

def test_real_receipts_and_strict_future_open(tmp_path):
    r,b,at=prediction(tmp_path)
    assert pd.Timestamp(r['record']['received_at'])==at-pd.Timedelta(minutes=1)
    assert pd.Timestamp(r['record']['earliest_executable_at'])>at
    assert next_open(at.floor('D'))==at.floor('D')+DAY
    assert sum(a['target_weight'] for a in r['record']['assets'])<=1

@pytest.mark.parametrize('fault',['stale','future_receipt','duplicate','gap','nat','timezone'])
def test_bad_source_suppresses_signal(tmp_path,fault):
    _,b,at=prediction(tmp_path)
    if fault=='stale': b=b[b.candle_ts<b.candle_ts.max()]
    if fault=='future_receipt': b['available_at']=at+DAY
    if fault=='duplicate': b=pd.concat([b,b.iloc[[0]]])
    if fault=='gap': b=b[b.candle_ts!=b.candle_ts.max()-10*DAY]
    if fault=='nat': b.loc[0,'available_at']=pd.NaT
    if fault=='timezone': b['candle_ts']=b.candle_ts.dt.tz_convert('US/Pacific')
    with pytest.raises(ValueError): latest_rule(b,at)

def test_transaction_race_idempotent_and_corrections(tmp_path):
    path=tmp_path/'ledger.jsonl'
    def add(_): return append_unique(path,'same',{'eligible_clean':True},'v1')[1]
    with ThreadPoolExecutor(max_workers=6) as pool: assert sum(pool.map(add,range(12)))==1
    old=Ledger(path).rows()[0]
    new,added=append_unique(path,'same',{'eligible_clean':True},'v2',True)
    assert added and new['record']['supersedes']==old['hash']
    assert new['record']['classification']=='CORRECTION_BACKFILL'
    assert not new['record']['eligible_clean']
    assert Ledger(path).verify()==2

def test_chain_corruption_refused(tmp_path):
    r,b,at=prediction(tmp_path)
    path=tmp_path/'p.jsonl'; path.write_text(path.read_text().replace('fixture-hash','damaged'))
    with pytest.raises(ValueError,match='chain'): append_unique(path,'other',{},'x')

@pytest.mark.parametrize('h',[14,28])
def test_actual_elapsed_time_and_price_receipt_boundary(tmp_path,h):
    r,b,at=prediction(tmp_path); start=pd.Timestamp(r['record']['earliest_executable_at'])
    future=bars(440)
    # Future fixture prices exist in memory, but have not arrived as of maturity.
    early=start+h*DAY-pd.Timedelta(seconds=1)
    received=future[future.candle_ts+DAY<=early].copy()
    received['available_at']=early
    s=settle([r],received,early,tmp_path/'s.jsonl')
    assert not any(x['record']['horizon']==h for x in Ledger(tmp_path/'s.jsonl').rows())
    when=start+(h+1)*DAY+pd.Timedelta(hours=3)
    received=future[future.candle_ts+DAY<=when].copy(); received['available_at']=when
    s=settle([r],received,when,tmp_path/'s.jsonl')
    labels=[x['record'] for x in Ledger(tmp_path/'s.jsonl').rows() if x['record']['horizon']==h]
    assert len(labels)==2
    for label in labels:
        assert label['actual_return']==pytest.approx(label['end_open']/label['start_open']-1)
    count=Ledger(tmp_path/'s.jsonl').verify()
    settle([r],received,when,tmp_path/'s.jsonl')
    assert Ledger(tmp_path/'s.jsonl').verify()==count

def test_pending_is_not_zero_return(tmp_path):
    r,b,at=prediction(tmp_path)
    s=settle([r],b,at,tmp_path/'s.jsonl')
    assert s['mature_labels']==0 and s['pending_labels']==4
    assert s['earliest_maturity'] and 'actual_return' not in json.dumps(s)
    assert independent_windows([r],[])==0

def test_no_two_asset_double_count_and_no_missing_observation_days():
    start=pd.Timestamp('2021-01-01',tz='UTC'); predictions=[]; settlements=[]
    for i in range(56):
        p=dict(eligible_clean=True,generated_at=str(start+(i-1)*DAY+pd.Timedelta(hours=2)),
               earliest_executable_at=str(start+i*DAY))
        predictions.append(dict(hash=str(i),record=p))
        for pair in ['ETHUSD','XBTUSD']:
            settlements.append(dict(record=dict(key=f'{i}:{pair}:28')))
    assert independent_windows(predictions,settlements)==2
    assert independent_windows(predictions[:10]+predictions[11:],settlements)==1

def test_actual_rule_exporter_truncation_uses_no_future_gap():
    from quant_platform.features.registry import compute_factors
    b=bars(); b=b[b.candle_ts!=pd.Timestamp('2020-08-01',tz='UTC')]
    f=pd.concat([compute_factors(g).assign(pair=p) for p,g in b.groupby('pair')])
    full=historical_signals(f)
    for cutoff in [pd.Timestamp('2020-07-31',tz='UTC'),pd.Timestamp('2021-01-01',tz='UTC')]:
        cut=historical_signals(f[f.candle_ts<=cutoff])
        for pair in full:
            pd.testing.assert_frame_equal(full[pair][full[pair].candle_ts<=cutoff].reset_index(drop=True),cut[pair])

def test_network_guard_used_by_processing():
    import socket
    with offline_guard(),pytest.raises(PermissionError): socket.getaddrinfo('api.kraken.com',443)

def test_superseded_observation_is_not_clean_evidence(tmp_path):
    r,b,at=prediction(tmp_path)
    correction=dict(r['record']); correction.pop('key'); correction.pop('content_hash')
    append_unique(tmp_path/'p.jsonl','fixture-day',correction,'revised',True)
    future=bars(440); when=future.candle_ts.max()+DAY
    future['available_at']=when
    s=settle(Ledger(tmp_path/'p.jsonl').rows(),future,when,tmp_path/'s.jsonl')
    assert s['mature_labels']==0
    assert independent_windows(Ledger(tmp_path/'p.jsonl').rows(),[])==0

def test_decode_discards_uncommitted_last_row():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_rule_forward import decode_public
    t=pd.Timestamp('2026-11-12',tz='UTC')
    rows=[[int((t+i*DAY).timestamp()),'100','102','98','101','100','10',3] for i in range(3)]
    raw=json.dumps(dict(error=[],result=dict(X=rows,last=0))).encode()
    r=decode_public(raw,'ETHUSD',t+4*DAY)
    assert len(r)==2 and r.candle_ts.max()==t+DAY
