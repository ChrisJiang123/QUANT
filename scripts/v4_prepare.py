"""Freeze hypotheses, split and preregistration BEFORE importing any model code."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_platform.research.governance import write_json,digest_file,offline_guard,now,git_sha,Ledger
from quant_platform.data.pit import load_snapshot,split_table
from quant_platform.features.registry import FactorRegistry,compute_factors,PRIMARY,quality_report
from v4_bootstrap import ROOT

def prepare():
    ex=ROOT/'experiments'; out=ex/'v4'
    if (out/'registration.json').exists():
        r=json.loads((out/'registration.json').read_text())
        for path,sha in r['files'].items():
            if digest_file(ROOT/path)!=sha:raise ValueError('registration changed')
        lineage=json.loads((out/'data_lineage.json').read_text())
        for path,sha in lineage['files'].items():
            if digest_file(ROOT/path)!=sha:raise ValueError('snapshot source changed')
        print('PREPARATION_RESUME_VERIFIED');return
    cfg=json.loads((ROOT/'configs/v4.json').read_text())
    hypotheses='''# HYPOTHESES V4 — registered before modelling
H1 Trend persistence: mom7/14/28 and trend120 direction; test DEV HAC MAE and costed trend versus hold. Fail on no costed improvement.
H2 Volatility predictability: vol7/28, downside28, ATR14 size exposure; compare s-only against s+vol target with fixed parameters. Fail on no same-period costed improvement.
H3 Crowding: funding and basis rolling z/ranks; only filters, never direction. Same-period L2 add/remove comparison. Missing history => INCONCLUSIVE.
H4 Scheduled risk: calendar announced_at <= decision_at; event filter, never direction. No historical publication proof => masked and INCONCLUSIVE.
H5 Attention extremes: Wikipedia/Fear & Greed exploratory only, excluded from G1-G3. Missing history => INCONCLUSIVE.
No LLM outputs used as historical features. Historical absent data never synthesized.
'''
    hp=ex/'HYPOTHESES_V4.md'
    if not hp.exists(): hp.write_text(hypotheses,encoding='utf-8')
    with offline_guard():
        b,lineage=load_snapshot(ROOT)
        write_json(out/'data_lineage.json',lineage,immutable=True)
        splits=split_table(str(b.candle_ts.min().date()),str(b.candle_ts.max().date()))
        write_json(ex/'SPLITS_V4.json',splits,immutable=True)
        b.to_parquet(out/'bars.parquet',index=False)
        frames=[]
        for pair,g in b.groupby('pair'):
            f=compute_factors(g); f['pair']=pair
            for h in [14,28]:
                f[f'y{h}']=f.close.shift(-h)/f.close-1
                f[f'label_available{h}']=f.candle_ts+__import__('pandas').Timedelta(days=h+1)
            frames.append(f)
        import pandas as pd
        f=pd.concat(frames,ignore_index=True)
        f['decision_at']=f.candle_ts+pd.Timedelta(days=1)
        f.to_parquet(out/'features.parquet',index=False)
        write_json(out/'factor_registry.json',FactorRegistry().entries(),immutable=True)
        write_json(out/'factor_quality/metrics.json',quality_report(f),immutable=True)
    pre=ex/'PREREGISTRATION_V4.md'
    if not pre.exists():
        text=f'''# V4 preregistration
Registered at {now()}; code {git_sha(ROOT)}.
Hypotheses SHA256: {digest_file(hp)}
Splits SHA256: {digest_file(ex/'SPLITS_V4.json')}
Configuration SHA256: {digest_file(ROOT/'configs/v4.json')}
Frozen configuration below defines the entire initial search space: ONE configuration per model family and ONE decision configuration; SIDGTN five seeds; no validation-driven search. Search caps 30/family, 12/decision. Every fit/failure recorded before validation metrics.
{json.dumps(cfg,indent=2)}
First validation starts 2016-11-01; expanding annual folds through DEV end. Training cutoff 28 days before validation; training labels must mature before cutoff. Inner early-stop segment is last 180 calendar days with a further 28-day purge; scalers fit inner training only. Embargo 28 days after a validation segment before its labels become eligible to a later training cutoff. No label overlap across train/validation boundaries.
Targets raw 14/28-day simple returns, no volatility normalization. Uniform sample weights, overlapping samples explicitly counted. Feature lookback 28, normalization fit train only, missing primary windows excluded; no imputation. Price stream daily log return, high/low range, log1p volume; train standardization. Fixed pinball losses for q10/q50/q90; BCE direction and future realized volatility heads, weights in config. No calibration fitting on evaluation data.
G1: both horizons MAE skill versus zero >0; one-sided DM HAC lag h-1 p<0.05, Holm correction across tested family/horizons.
G2: BASE Sharpe delta versus strongest simple baseline >=0.20; 28-day block bootstrap 90% lower CI >0; DSR >=0.95, trial count from ledger. Paired blocks across correlated assets; annualization 365.
G3: STRESS delta positive; five-seed median G2 passes, worst seed nonnegative; majority state segments nonnegative.
G4: fresh independent 28-day windows >=12, direction consistent. Fewer windows => terminal INCONCLUSIVE_DATA. G1-G3 failure => overall NO_GO_ALPHA. Missing costed evidence never passes a gate.
Strongest simple: buy-hold, rule ensemble, Ridge/ElasticNet/LightGBM, selected on DEV only. SIDGTN tested separately relative to strongest simple ML. Rule windows 28/60/120, vol target 15%, max asset 50%, band 5%, schedule every 14 days from 2016-11-01, cash allowed, max 28-day trade holding with renewed entry next scheduled opportunity. Research limits are not authorized live limits.
H3/H4 disabled until PIT data coverage/publication verified. H5 exploratory. No pretending forward-only data has ten-year history.
Monitoring: source outage >3d mask, price outage no prediction; chain break stop; 90d PSI >.25 DRIFT; >=12 independent windows and coverage deviation >.15 MISCALIBRATED; outside bootstrap 5%-95% DEGRADED; >=12 independent 28d windows with negative costed excess => retire to rule. Changes start new track.
'''
        pre.write_text(text,encoding='utf-8')
    write_json(out/'registration.json',dict(schema_version=1,registered_at=now(),
                 files={str(p.relative_to(ROOT)):digest_file(p) for p in [hp,pre,ex/'SPLITS_V4.json',ROOT/'configs/v4.json']}),immutable=True)
    Ledger(out/'ledger.jsonl').append(dict(kind='REGISTERED',registration_sha256=digest_file(out/'registration.json'),git_sha=git_sha(ROOT)))
    write_json(out/'preparation/metrics.json',dict(schema_version=1,rows=len(b),feature_rows=len(f),primary_factors=len(PRIMARY),
                fresh_windows=splits['FRESH_HOLDOUT']['independent_28d_windows'],gaps=len(lineage['gaps'])),immutable=True)
    print('PREPARED; preregistration frozen; no model evaluated')
if __name__=='__main__': prepare()
