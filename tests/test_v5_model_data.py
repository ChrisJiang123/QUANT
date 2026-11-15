import copy
import numpy as np
import pandas as pd
import pytest
import torch
from quant_platform.v5.model import V5SIDGTN,loss
from quant_platform.v5.data import future_open_targets,join_asof,EventCalendarAdapter
from quant_platform.v5.common import CFG,OUT,read,registered

torch.set_num_threads(4)

def batch(n=4):
    torch.manual_seed(123)
    return dict(price=torch.randn(n,60,3),factors=torch.randn(n,8),aux=torch.randn(n,8),aux_mask=torch.ones(n,8,dtype=torch.bool),
        nodes=torch.randn(n,8,8),node_mask=torch.ones(n,8,dtype=torch.bool),adjacency=torch.ones(n,8,8,dtype=torch.bool),
        sentiment=torch.randn(n,7,4),sentiment_mask=torch.ones(n,7,dtype=torch.bool),asset_id=torch.arange(n)%2,
        annual_anchor=torch.ones(n)*.6,target=torch.randn(n,5)*.1,target_vol=torch.ones(n,5)*.7,label_mask=torch.ones(n,5,dtype=torch.bool))

@pytest.mark.parametrize('variant',['core','factor','graph','sentiment'])
def test_all_five_horizons_finite_ordered_and_branch_gradients(variant):
    model=V5SIDGTN(variant=variant);b=batch();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    value,o,_=loss(model,b,CFG['loss']);value.backward();opt.step()
    assert o['quantiles'].shape==(4,5,3)
    assert (o['quantiles'][:,:,0]<=o['quantiles'][:,:,1]).all()
    assert (o['quantiles'][:,:,1]<=o['quantiles'][:,:,2]).all()
    assert (o['tail']<=o['quantiles'][:,:,0]).all()
    assert (o['volatility']>=.10).all() and (o['direction_score'].abs()<=1).all()
    needed=['price_conv','factor']+(['auxiliary'] if variant!='core' else [])+(['edge_attention'] if variant in {'graph','sentiment'} else [])+(['sentiment'] if variant=='sentiment' else [])
    for prefix in needed:
        assert sum(float(p.grad.abs().sum()) for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None)>0
        assert sum(float((p.detach()-before[k]).abs().sum()) for k,p in model.named_parameters() if k.startswith(prefix))>0

def test_all_missing_modalities_exact_fallback_and_fake_neutral_not_learned():
    b=batch();b['aux_mask'][:]=False;b['node_mask'][:]=False;b['sentiment_mask'][:]=False
    model=V5SIDGTN(variant='sentiment').eval();a=model(b)
    polluted=copy.deepcopy(b)
    for k in ['aux','nodes','sentiment']:polluted[k][:]=float('nan')
    z=model(polluted)
    for k in ['quantiles','direction_score','volatility','tail']:
        assert torch.isfinite(z[k]).all();torch.testing.assert_close(a[k],z[k],rtol=0,atol=0)
    assert not z['modality_mask'][:,2:].any() and (z['fusion_weights'][:,2:]==0).all()
    assert z['graph_attention'].abs().sum()==0

def test_dynamic_fusion_cannot_enable_without_evidence_and_masked_execution_works():
    with pytest.raises(PermissionError):V5SIDGTN(dynamic_fusion=True)
    model=V5SIDGTN(variant='sentiment',dynamic_fusion=True,fusion_qualified=True)
    b=batch();value,o,_=loss(model,b,CFG['loss']);value.backward()
    assert model.cross.in_proj_weight.grad.abs().sum()>0
    assert torch.isfinite(o['quantiles']).all()

def test_pit_join_uses_receipt_not_event_date_and_masks_stale():
    t=pd.Timestamp('2020-01-01',tz='UTC')
    events=pd.DataFrame(dict(available_at=[t+pd.Timedelta(days=2)],value=[99.]))
    times=pd.date_range(t,periods=7,freq='D')
    x,m=join_asof(times,events,['value'],max_age_days=1)
    assert not m[:2].any() and m[2:4].all() and not m[4:].any()
    assert x[0,0]==0 and x[2,0]==99

def test_targets_are_future_executable_opens_and_gaps_mask_labels():
    t=pd.date_range('2020-01-01',periods=45,tz='UTC');p=np.arange(100.,145.)
    f=pd.DataFrame(dict(candle_ts=t,open=p))
    y,v,m,index=future_open_targets(f,[1,3,7,14,28])
    assert y[0,0]==pytest.approx(103/102-1)
    assert y[0,4]==pytest.approx(130/102-1)
    f=f[f.candle_ts!=t[10]]
    y,v,m,index=future_open_targets(f,[1,28]);assert not m[0,1]

def test_calendar_without_announced_vintage_refused():
    with pytest.raises(ValueError):EventCalendarAdapter(pd.DataFrame())

def test_real_dataset_branch_coverage_and_registered_source():
    registered();manifest=read(OUT/'data/manifest.json')
    raw=np.load(OUT/'data/dataset.npz',allow_pickle=False)
    assert manifest['samples']==len(raw['price'])>5000
    assert raw['nodes'].shape[1:]==(8,8)
    assert raw['aux_mask'].any() and raw['node_mask'].sum()>0
    assert raw['sentiment_mask'].sum()==0
    assert all(x==False for x in raw['aux_mask'][:,[3,7]].ravel())
    assert sorted(manifest['node_mapping'].values())==list(range(8))

def test_real_completed_fixture_is_not_4090_or_financial_evidence():
    receipt=read(OUT/'fixture_job.json');folder=__import__('pathlib').Path(__file__).resolve().parents[1]/receipt['job_root']
    for variant in ['core','factor','graph']:
        m=read(folder/'results/wf2022'/variant/'seed11/metrics.json')
        assert m['device']=='cpu' and m['epochs_completed']==1
        if variant!='core':assert m['real_aux_samples']>0 and m['gradient_norms']['auxiliary']>0
        if variant=='graph':assert m['real_graph_samples']>0 and m['gradient_norms']['edge_attention']>0
