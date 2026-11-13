import json
import numpy as np
import pandas as pd
import pytest
import torch
from quant_platform.models.modalities import SentimentEncoder,GraphEncoder,DynamicFusion
from quant_platform.features.mining import FactorDSL
from quant_platform.data.prospective import ProspectiveSource,calendar_features
from quant_platform.research.observation import append_forward,append_shadow,monitor,psi

def test_optional_modalities_masks_graph_and_fallback():
    s=SentimentEncoder();o=s(torch.full((2,3,4),float('nan')),torch.zeros(2,3,dtype=torch.bool));assert torch.equal(o,torch.zeros_like(o))
    g=GraphEncoder();nodes,attention=g(torch.randn(2,2,8),torch.ones(2,2,2,dtype=torch.bool),torch.tensor([[True,True],[False,False]]))
    assert nodes.shape==(2,2,24) and torch.isfinite(nodes).all() and nodes[1].abs().sum()==0
    f=DynamicFusion();values=torch.randn(2,4,24);mask=torch.tensor([[True,False,False,False],[True,True,False,False]])
    out,w=f(values,mask,torch.zeros(2,3));torch.testing.assert_close(out[0],values[0,0]);assert w[:,2:].sum()==0

@pytest.mark.parametrize('expression',["__import__('os')","close.shift(-1)","open('/tmp/a')","close[0]",'lag(close,-1)',
       'mean(close,1000000)','close.__class__','[x for x in close]','close ** 100000000','eval(close)','lambda:close'])
def test_dsl_dangerous_or_future_expression_refused(expression):
    with pytest.raises((ValueError,SyntaxError)):FactorDSL().parse(expression)

def test_dsl_safe_candidate_is_deduplicated_pending_approval(tmp_path):
    d=FactorDSL();f=pd.DataFrame({'close':np.arange(100.)+1})
    result=d.evaluate('close / lag(close,7) - 1',f)
    assert result.iloc[0:7].isna().all()
    first=d.record_candidate('close/lag(close,7)-1',f,tmp_path/'candidates.jsonl')
    second=d.record_candidate('close / lag(close, 7) - 1',f,tmp_path/'candidates.jsonl')
    assert first['status']=='PENDING_HUMAN_APPROVAL' and second['duplicate']

def test_source_delay_outage_mask_and_calendar_future_injection():
    t=pd.Timestamp('2020-01-01',tz='UTC');dates=pd.date_range(t,periods=5)
    source=ProspectiveSource('wikipedia');frame=pd.DataFrame({'observed_at':[t],'value':[10]})
    joined=source.join(dates,frame);assert joined.value.iloc[:3].isna().all() and joined.value.iloc[3]==10
    assert source.join(dates,None).source_mask.all()
    event={'announced_at':(t+pd.Timedelta(days=2)).isoformat(),'event_at':(t+pd.Timedelta(days=1)).isoformat(),'scheduled':True}
    assert calendar_features([event],t)['event_coefficient']==1

def test_forward_daily_chain_and_shadow_separation(tmp_path):
    record=dict(track_id='test',model_hash='a'*64,data_hash='b'*64,predicted_at='2020-01-01T01:00:00Z',decision_at='2020-01-01T00:00:00Z',predictions={})
    append_forward(tmp_path/'forward.jsonl',record)
    with pytest.raises(ValueError):append_forward(tmp_path/'forward.jsonl',record)
    with pytest.raises(ValueError):append_forward(tmp_path/'other.jsonl',{**record,'source':'llm_shadow'})
    with pytest.raises(ValueError):append_shadow(tmp_path/'shadow.jsonl',dict(backend='paid_remote',temperature=0),record['predicted_at'])
    with pytest.raises(ValueError):append_shadow(tmp_path/'shadow.jsonl',dict(backend='local_ollama',temperature=0,news=[{'published_at':'2030'}]),record['predicted_at'])

def test_monitoring_never_enables_trading():
    m=monitor(4,False,False,.5,12,.2,-.1,True)
    assert {'STOP_CHAIN','MASK_SOURCE','DRIFT','MISCALIBRATED','RETIRE_TO_RULE'}.issubset(m['status'])
    assert not m['live_enabled'] and not m['order_enabled']
    assert psi(np.zeros(100),np.ones(100)) is None
