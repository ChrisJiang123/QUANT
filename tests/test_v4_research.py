import ast,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from quant_platform.models.sidgtn import SIDGTN,multitask_loss
from quant_platform.research.governance import digest_file,ensure_research,offline_guard,Ledger
from quant_platform.features.registry import compute_factors,PRIMARY
from quant_platform.research.statistics import dm_hac,paired_bootstrap,holm
from v4_engine_metrics import reconciliation
from test_v4_contracts import bars
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/v4'

def test_core_no_trading_sdk_imports_or_packages():
    import importlib.util
    banned={'ccxt','freqtrade','alpaca','ib_insync','ibapi'}
    for file in (ROOT/'src').rglob('*.py'):
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            assert not banned.intersection(n.split('.')[0] for n in names),str(file)
    assert not any(importlib.util.find_spec(n) for n in banned)

def test_no_credentials_and_execution_flags():
    cfg=json.loads((ROOT/'configs/v4.json').read_text());ensure_research(cfg)
    for field in ['live_enabled','order_enabled']:
        bad=dict(cfg);bad[field]=True
        with pytest.raises(PermissionError):ensure_research(bad)
    with pytest.raises(PermissionError):ensure_research({**cfg,'exchange':{'api_key':'forbidden'}})

def test_model_real_gradients_capacity_quantiles_and_cpu_gpu():
    torch.manual_seed(11);m=SIDGTN();m.eval()
    p=torch.randn(4,28,3);x=torch.randn(4,len(PRIMARY));a=torch.tensor([0,1,0,1]);o=m(p,x,a)
    assert sum(v.numel() for v in m.parameters())<100000
    assert o['quantiles'].shape==(4,2,3)
    assert (o['quantiles'][:,:,0]<=o['quantiles'][:,:,1]).all()
    assert (o['quantiles'][:,:,1]<=o['quantiles'][:,:,2]).all()
    loss=multitask_loss(o,torch.zeros(4,2),torch.ones(4,2));loss.backward()
    assert m.price[0].weight.grad.abs().sum()>0 and m.factor[0].weight.grad.abs().sum()>0
    if torch.cuda.is_available():
        m.zero_grad();m.cuda();g=m(p.cuda(),x.cuda(),a.cuda())
        torch.testing.assert_close(o['quantiles'],g['quantiles'].cpu(),atol=1e-5,rtol=1e-4)

def test_future_observation_cannot_change_factors_or_frozen_prediction():
    b=bars(400);f=compute_factors(b);altered=b.copy();altered.loc[301:,'close']*=10
    altered.loc[301:,'high']=altered.loc[301:,'close']+1;altered.loc[301:,'low']=altered.loc[301:,'close']-1
    g=compute_factors(altered)
    np.testing.assert_array_equal(f[PRIMARY].iloc[300].to_numpy(),g[PRIMARY].iloc[300].to_numpy())
    torch.manual_seed(22);model=SIDGTN();model.eval();p=torch.zeros(1,28,3);aid=torch.zeros(1,dtype=torch.long)
    with torch.no_grad():
        first=model(p,torch.tensor(f[PRIMARY].iloc[300].to_numpy(),dtype=torch.float32)[None],aid)['quantiles']
        second=model(p,torch.tensor(g[PRIMARY].iloc[300].to_numpy(),dtype=torch.float32)[None],aid)['quantiles']
    torch.testing.assert_close(first,second,atol=0,rtol=0)

def test_trained_scalers_purge_and_label_maturity():
    paths=list((OUT/'training').glob('dev_fold*/normalizer.json'));assert paths
    for p in paths:
        a=json.loads(p.read_text());start=pd.Timestamp(a['validation_start']);cut=pd.Timestamp(a['train_cutoff'])
        assert (start-cut).days>=28
        assert pd.Timestamp(a['train_label_max'])<cut
        assert pd.Timestamp(a['fit_max_decision'])<cut-pd.Timedelta(days=28)
    forecasts=pd.read_parquet(OUT/'dev_predictions.parquet')
    evaluated=forecasts[forecasts.metric_eligible]
    assert ((evaluated.decision_at+pd.to_timedelta(evaluated.horizon,unit='D'))<evaluated.validation_end).all()

def test_all_original_data_and_negative_results_unchanged():
    before=json.loads((OUT/'input_manifest.json').read_text())['files']
    for name,row in before.items():
        if name.startswith(('data/','experiments/price/','src/quant_crypto/','tests/')) or name=='TRAINING_REPORT.md':
            assert digest_file(ROOT/name)==row['sha256'],name

def test_raw_storage_write_refused_and_corruption_detected(tmp_path):
    from quant_crypto.storage import immutable_write,verify
    p=tmp_path/'raw.csv';immutable_write(p,b'original')
    with pytest.raises((PermissionError,OSError)):p.write_bytes(b'changed')
    with pytest.raises(ValueError):immutable_write(p,b'changed')
    with pytest.raises(ValueError):verify(p,'0'*64)

def test_engine_conversion_identical_and_gaps_not_filled():
    canonical=pd.read_parquet(OUT/'bars.parquet')
    for pair in ['XBTUSD','ETHUSD']:
        name='BTC_USD' if pair=='XBTUSD' else 'ETH_USD'
        actual=pd.read_feather(OUT/f'freqtrade_v2/data/{name}-1d.feather')
        source=canonical[canonical.pair==pair]
        assert len(actual)==len(source)
        np.testing.assert_array_equal(actual.close.to_numpy(),source.close.to_numpy())
        assert actual.date.tolist()==source.candle_ts.tolist()

def test_engine_weight_reconciliation():
    r=reconciliation(OUT/'freqtrade_v2/DEV/rule/BASE')
    assert r['passed'],r

def test_hac_overlap_correction_and_holm():
    r=dm_hac(np.ones(100)*.01+np.sin(np.arange(100))*.001,28)
    assert r['hac_lag']==27 and r['p_one_sided']<.05
    assert holm([.01,.04])==[.02,.04]
    ci=paired_bootstrap(np.zeros(20),np.zeros(20));assert ci['lower90'] is None

def test_preregistration_predates_first_validation_and_chain_valid():
    reg=json.loads((OUT/'registration.json').read_text());l=Ledger(OUT/'ledger.jsonl');assert l.verify()>0
    starts=[r['timestamp'] for r in l.rows() if r['record'].get('kind')=='TRIAL_STARTED']
    assert min(starts)>reg['registered_at']
    assert (ROOT/'experiments/HYPOTHESES_V4.md').stat().st_mtime<min(p.stat().st_mtime for p in (OUT/'training').glob('*/metrics.json'))
