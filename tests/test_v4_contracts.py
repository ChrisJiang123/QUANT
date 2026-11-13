"""Contract and refusal tests written before V4 implementations. Synthetic only."""
import json
import socket
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from quant_platform.research.governance import Ledger, consume_final, freeze, offline_guard, digest_file
from quant_platform.features.registry import compute_factors, PRIMARY
from quant_platform.data.pit import pit_join, validate_bars, split_table
from quant_platform.adapters.crypto.offline_freqtrade import authorize_cli
from quant_platform.strategy.decision import target_weights

def bars(n=350):
    ts=pd.date_range('2020-01-01',periods=n,tz='UTC')
    c=100+np.sin(np.arange(n)/10)+np.arange(n)/10
    return pd.DataFrame(dict(candle_ts=ts,open=c,high=c+1,low=c-1,close=c,volume=100.,
                             observed_at=ts+pd.Timedelta(days=1),available_at=ts+pd.Timedelta(days=1)))

def test_offline_rejects_network():
    with offline_guard():
        with pytest.raises(PermissionError): socket.create_connection(('example.com',443))

def test_ledger_chain_tampering(tmp_path):
    l=Ledger(tmp_path/'ledger.jsonl'); l.append({'x':1}); l.append({'x':2}); assert l.verify()==2
    s=l.path.read_text().replace('"x": 1','"x": 9'); l.path.write_text(s)
    with pytest.raises(ValueError): l.verify()

def test_final_once_and_manifest_rejects_changes(tmp_path):
    source=tmp_path/'code.py'; source.write_text('version 1')
    manifest=freeze([source]); marker=tmp_path/'final.json'
    consume_final(marker,manifest); assert marker.exists()
    with pytest.raises(PermissionError): consume_final(marker,manifest)
    consume_final(marker,manifest,audit_rerun=True)
    source.write_text('version 2')
    with pytest.raises(ValueError): consume_final(marker,manifest,audit_rerun=True)
    with pytest.raises(PermissionError): consume_final(marker,freeze([source]),audit_rerun=True)

@pytest.mark.parametrize('cmd',['trade','webserver','download-data','hyperopt','--help','backtesting;trade'])
def test_freqtrade_refuses_unsafe(cmd):
    with pytest.raises(PermissionError): authorize_cli([cmd])

def test_freqtrade_requires_fee_and_forbids_injected_config():
    with pytest.raises(PermissionError): authorize_cli(['backtesting'])
    with pytest.raises(PermissionError): authorize_cli(['backtesting','--fee','0.0085','--api-key','x'])
    authorize_cli(['backtesting','--fee','0.0085'])

def test_point_in_time_future_injection():
    t=pd.Timestamp('2020-01-03',tz='UTC')
    values=pd.DataFrame({'available_at':[t-pd.Timedelta(days=1),t+pd.Timedelta(days=1)],'value':[1,999]})
    assert pit_join(pd.DatetimeIndex([t]),values)['value'].iloc[0]==1

def test_truncated_factor_recompute():
    b=bars(); allf=compute_factors(b)
    truncated=compute_factors(b.iloc[:300])
    np.testing.assert_allclose(allf[PRIMARY].iloc[299],truncated[PRIMARY].iloc[-1])

def test_gap_remains_missing_and_masks_windows():
    b=bars().drop(index=299).reset_index(drop=True)
    f=compute_factors(b)
    assert f.loc[f.candle_ts==pd.Timestamp('2020-10-27',tz='UTC'),'mom28'].isna().all()
    bad=bars(); bad.loc[2,'close']=-1
    with pytest.raises(ValueError): validate_bars(bad)

def test_weights_long_cash_no_leverage_and_shadow_rejected():
    w=target_weights(np.array([1.,1.]),np.array([.1,.1]))
    assert np.all(w>=0) and np.all(w<=.5) and w.sum()<=1
    with pytest.raises(ValueError): target_weights(np.array([1.,1.]),np.array([.1,.1]),source='llm_shadow')

def test_split_fresh_empty_not_fabricated():
    s=split_table('2013-10-06','2026-09-30')
    assert s['FRESH_HOLDOUT']['independent_28d_windows']==0
    assert s['TOUCHED_AUDIT']['label']=='PREVIOUSLY_CONSUMED'
