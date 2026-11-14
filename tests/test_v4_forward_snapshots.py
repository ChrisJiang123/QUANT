import json
from pathlib import Path
import numpy as np,pandas as pd
import pytest
from quant_platform.data.forward import read_forward_bars,latest_inputs,matured_labels
from quant_platform.research.governance import digest_file
from quant_platform.features.registry import PRIMARY
from test_v4_contracts import bars
ROOT=Path(__file__).resolve().parents[1]

def test_forward_snapshot_hash_refuses_tampering(tmp_path):
    b=pd.concat([bars(350).assign(pair=p) for p in ['XBTUSD','ETHUSD']],ignore_index=True)
    p=tmp_path/'bars.parquet';b.to_parquet(p,index=False)
    (tmp_path/'manifest.json').write_text(json.dumps({'schema_version':1,'files':{'bars.parquet':digest_file(p)}}))
    assert len(read_forward_bars(p,pd.Timestamp('2021-01-01',tz='UTC')))==700
    p.write_bytes(p.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):read_forward_bars(p,pd.Timestamp('2021-01-01',tz='UTC'))

def test_forward_snapshot_future_observation_rejected(tmp_path):
    b=pd.concat([bars(350).assign(pair=p) for p in ['XBTUSD','ETHUSD']],ignore_index=True)
    p=tmp_path/'bars.parquet';b.to_parquet(p,index=False)
    with pytest.raises(ValueError,match='unavailable'):read_forward_bars(p,pd.Timestamp('2020-01-02',tz='UTC'),False)

def test_new_snapshot_input_pipeline_matches_frozen_definition():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from v4_train import dataset
    b=pd.read_parquet(ROOT/'experiments/v4/bars.parquet');as_of=b.available_at.max()+pd.Timedelta(hours=1)
    latest,p,aid=latest_inputs(b,as_of);m,old,_=dataset();rows=m.groupby('pair').tail(1).sort_values('pair')
    np.testing.assert_allclose(p,old[rows.sample_id.to_numpy()],atol=0,rtol=0)
    np.testing.assert_allclose(latest[PRIMARY],rows[PRIMARY],atol=1e-12,rtol=1e-12)

def test_matured_labels_need_new_available_prices():
    t=pd.Timestamp('2020-01-01',tz='UTC');pred=[{'pair':'XBTUSD','decision_at':str(t),'anchor_close':100.,'horizons':[14,28]}]
    future=pd.DataFrame({'pair':['XBTUSD','XBTUSD'],'available_at':[t+pd.Timedelta(days=14),t+pd.Timedelta(days=28)],'close':[110.,120.]})
    assert not matured_labels(pred,future,t)
    assert len(matured_labels(pred,future,t+pd.Timedelta(days=14)))==1
    labels=matured_labels(pred,future,t+pd.Timedelta(days=28));assert len(labels)==2 and labels[1]['actual_return']==pytest.approx(.2)
