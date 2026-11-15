import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from quant_platform.v5.common import ROOT,OUT,CFG,read,registered
from quant_platform.research.governance import digest_file
from quant_platform.v5.data import PriceAdapter,SpecificationAdapter
from quant_platform.v5.engine import Engine
from quant_platform.v5.book import Book
from quant_platform.v5.instruments import InstrumentSpec

def test_registration_and_v4_frozen_bytes_are_preserved():
    registered()
    for p,h in read(OUT/'v4_preservation.json')['files'].items():assert digest_file(ROOT/p)==h
    frozen=read(ROOT/'experiments/v4/frozen.json')
    for p,h in frozen['files'].items():assert digest_file(p)==h

def test_actual_gpu_training_and_return_hashes_all_seeds_and_branches():
    v=read(OUT/'gpu_verification.json');assert v['state']=='VERIFIED' and v['fits']==60
    assert '4090' in v['environment']['gpu']
    assert {(x['variant'],x['seed']) for x in v['checks']}=={(a,s) for a in CFG['model']['variants'] for s in CFG['seeds']}
    returned=ROOT/v['return_root'];manifest=read(returned/'return_manifest.json')
    for p,h in manifest['files'].items():assert digest_file(returned/p)==h
    assert max(x['max_cpu_gpu_absolute_error'] for x in v['checks'])<2e-4
    assert all(x['real_sentiment_samples']==0 for x in v['checks'])

def test_real_purged_folds_and_train_only_scaler_boundaries():
    receipt=read(OUT/'gpu_job.json');job=read(ROOT/receipt['job_root']/'job.json');meta=pd.read_parquet(OUT/'data/samples.parquet')
    for f in job['folds']:
        cutoff=pd.Timestamp(f['train_cutoff']);end=pd.to_datetime(meta.label_end28,utc=True)
        assert (end.iloc[f['fit']]<cutoff-pd.Timedelta(days=208)).all()
        assert (end.iloc[f['stop']]<cutoff).all()
        assert not set(f['fit'])&set(f['predict'])
    assert '2025-01-01' in job['folds'][-1]['evaluation_start']

def test_price_identity_and_spec_vintages_cannot_be_omitted():
    with pytest.raises(ValueError):PriceAdapter(pd.DataFrame(),'missing','REAL_US_PERP',kind='mark')
    with pytest.raises(ValueError):SpecificationAdapter(pd.DataFrame())

def test_maintenance_blocks_fills_but_does_not_stop_accounting():
    t=pd.Timestamp('2024-01-01',tz='UTC');s=InstrumentSpec('s:X','X',1,.01,fixed_fee=0)
    b=Book({'X':s},initial=7000,perp_bps=0);e=Engine(b,CFG)
    e.step(t,{'X':dict(open=100,high=100,low=100,close=100)},targets={('trend','X'):3},maintenance_window=True)
    assert b.physical['X'].quantity==0
    assert any(x['kind']=='MAINTENANCE_WINDOW' for x in b.events)
    assert len(e.history)==1

def test_actual_weight_inference_is_separate_from_forward_validation():
    result=read(ROOT/read(OUT/'observation_latest.json')['result'])
    assert result['live_enabled'] is False and result['order_enabled'] is False
    assert len(result['model_inference'])==30
    assert len(result['rule_track'])==2 and all(x['source']=='ACTUAL_RULE_CONTROLLER' for x in result['rule_track'])
    assert all(x['source']=='ACTUAL_VERIFIED_SIDGTN_CPU_INFERENCE' for x in result['model_inference'])
    assert result['qualified_forward_rows']==0 and result['clean_28d_windows']==0

def test_financial_reports_never_promote_consumed_final_or_us_scenarios():
    v=read(OUT/'validation.json');assert v['final']['status']=='REFUSED_NO_CLEAN_FINAL'
    assert v['final']['independent_clean_windows']==0
    assert v['final']['consumed_flag_sha256']==digest_file(ROOT/'experiments/FINAL_CONSUMED.json')
    assert all(not x['us_survival_verified'] and x['evidence']=='SCENARIO_ONLY' for x in v['all_metrics'].values())
    assert not v['dynamic_fusion_executed'] or v['dynamic_fusion_qualified']

def test_replay_resume_matches_uninterrupted_account(tmp_path):
    from quant_platform.v5.replay import replay
    prepared=pd.read_pickle(OUT/'replay_inputs.pkl')
    full=replay(tmp_path/'full',start='2022-09-28',end='2022-10-03',prepared=prepared)
    replay(tmp_path/'resume',start='2022-09-28',end='2022-10-03',prepared=prepared,stop_after=2)
    resumed=replay(tmp_path/'resume',start='2022-09-28',end='2022-10-03',prepared=prepared,resume=True)
    assert full['net_return']==pytest.approx(resumed['net_return'],abs=1e-12)
    assert read(tmp_path/'full/state/book.json')['events']==read(tmp_path/'resume/state/book.json')['events']

def test_research_cli_has_no_live_mode():
    import subprocess,sys
    p=subprocess.run([sys.executable,str(ROOT/'scripts/v5.py'),'live'],cwd=ROOT,capture_output=True)
    assert p.returncode==2

def test_replay_process_guard_is_active_before_any_cache_read(monkeypatch):
    import sys,socket,importlib.util
    spec=importlib.util.spec_from_file_location('v5_guard_test',ROOT/'scripts/v5_backtest.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def injected(*a,**kw):
        # Audit hook denies socket creation itself; no network is reached.
        socket.socket()
    monkeypatch.setattr(module.pd,'read_pickle',injected)
    with pytest.raises(PermissionError,match='OFFLINE_NO_EXTERNAL_CALLS'):
        module.task(dict(name='guard_test'),None,'unit_fixture')
