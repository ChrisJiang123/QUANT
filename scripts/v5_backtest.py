"""All preregistered comparisons, resumable per exact source/input fingerprint."""
import argparse,json,sys,pickle,concurrent.futures
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,CFG,read,registered
from quant_platform.v5.replay import inputs,replay
from quant_platform.research.governance import digest,digest_file,write_json,Ledger,now,offline_guard

def legacy():
    f=pd.read_parquet(ROOT/'experiments/v4/dev_predictions.parquet');f=f[f.model.str.startswith('sidgtn_seed')].copy()
    f=f[(f.candle_ts>=pd.Timestamp('2021-12-30',tz='UTC'))&(f.candle_ts<pd.Timestamp('2024-12-30',tz='UTC'))]
    # Re-time old frozen forecasts to V5's delayed, truly executable open.
    # Their old labels remain contaminated diagnostics; financial replay uses
    # the original available signal and V5's same conservative execution lag.
    f['execution_at']=f.candle_ts+pd.Timedelta(days=2);f['available_at']=f.candle_ts+pd.Timedelta(days=1,hours=6)
    f['seed']=f.model.str.extract(r'(\d+)$').astype(int);f['direction_score']=2*f.direction_probability-1
    f['annual_vol']=f.forecast_vol;f['median_score']=(f.prediction/f.forecast_vol/np.sqrt(f.horizon/365)).clip(-1,1)
    g=f.groupby(['seed','pair','execution_at','available_at'],as_index=False)[['direction_score','annual_vol','median_score']].mean()
    g['variant']='legacy_v4';return g

@offline_guard()
def task(spec,cache,fingerprint):
    folder=OUT/'backtests_v2'/spec['name']
    if (folder/'complete.json').exists():
        d=read(folder/'complete.json')
        if d['fingerprint']!=fingerprint:raise ValueError('completed comparison source changed; retain and use new experiment namespace')
        if digest_file(folder/'metrics.json')!=d['metrics_hash']:raise ValueError('metrics changed')
        return dict(name=spec['name'],status='RESUME_VERIFIED')
    prepared=pd.read_pickle(cache)
    args={k:v for k,v in spec.items() if k not in ['name','model_source']}
    if spec.get('model_source'):
        p=pd.read_parquet(OUT/('legacy_predictions.parquet' if spec['model_source']=='legacy' else 'model_predictions.parquet'))
        p=p[(p.seed==spec['seed'])&(p.variant==spec['variant'])]
        if 'fold' in p:p=p[p.fold!='final_dev_fit']
        args['predictions']=p
    m=replay(folder,prepared=prepared,**args)
    write_json(folder/'complete.json',dict(fingerprint=fingerprint,metrics_hash=digest_file(folder/'metrics.json'),model_prediction_hash=digest_file(OUT/('legacy_predictions.parquet' if spec.get('model_source')=='legacy' else 'model_predictions.parquet')) if spec.get('model_source') else None,completed_at=now(),spec=spec))
    return dict(name=spec['name'],status='COMPLETED',sharpe=m['sharpe'],survival=m['survival_pass'])

@offline_guard()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--smoke',action='store_true');parser.add_argument('--rules-only',action='store_true');args=parser.parse_args()
    registered()
    if not args.rules_only and read(OUT/'gpu_verification.json')['state']!='VERIFIED':raise ValueError('unverified models')
    cache=OUT/'replay_inputs.pkl'
    if not cache.exists():pd.to_pickle(inputs(),cache)
    if not args.rules_only:legacy().to_parquet(OUT/'legacy_predictions.parquet',index=False)
    sources={str(p.relative_to(ROOT)):digest_file(p) for p in list((ROOT/'src/quant_platform/v5').glob('*.py'))+[Path(__file__)]}
    fingerprint=digest(dict(sources=sources,registration=registered(),job_hash=read(OUT/'gpu_job.json')['job_hash'],prepared=digest_file(cache)))
    specs=[]
    for cost in CFG['costs']:
        for mode in ['buy_hold','long_only','asymmetric','filtered','carry','carry_only','tactical']:
            specs.append(dict(name=f'rule_{mode}_{cost}',mode=mode,cost=cost))
        for variant in CFG['model']['variants']:
            for seed in CFG['seeds']:
                for quadrant in (['rr','mr','rm','mm'] if cost=='BASE' else ['mm']):
                    specs.append(dict(name=f'{variant}_{seed}_{quadrant}_{cost}',mode='model',cost=cost,variant=variant,seed=seed,quadrant=quadrant,model_source='v5'))
                if cost=='BASE':specs.append(dict(name=f'{variant}_{seed}_median_BASE',mode='model',cost=cost,variant=variant,seed=seed,quadrant='mm',mapping='median',model_source='v5'))
    for seed in CFG['seeds']:
        for quadrant in ['rr','mr','rm','mm']:
            specs.append(dict(name=f'legacy_v4_{seed}_{quadrant}_BASE',mode='model',cost='BASE',variant='legacy_v4',seed=seed,quadrant=quadrant,model_source='legacy'))
        specs.append(dict(name=f'legacy_v4_{seed}_median_BASE',mode='model',cost='BASE',variant='legacy_v4',seed=seed,quadrant='mm',mapping='median',model_source='legacy'))
    for setting in ['funding_stress','margin_hike','favorable_first']:
        specs.append(dict(name=f'rule_filtered_{setting}',mode='filtered',cost='BASE',**{setting:True}))
    if args.smoke:specs=[dict(name='common_smoke',mode='filtered',cost='BASE',start='2022-01-01',end='2022-02-01')]
    if args.rules_only:specs=[s for s in specs if not s.get('model_source')]
    write_json(OUT/('backtest_smoke_plan.json' if args.smoke else 'rules_plan.json' if args.rules_only else 'backtest_plan.json'),dict(fingerprint=fingerprint,source_hashes=sources,trials=specs,created_at=now()))
    Ledger(OUT/'ledger.jsonl').append(dict(kind='BACKTEST_PLAN',fingerprint=fingerprint,trials=len(specs),fixture=args.smoke))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(task,s,cache,fingerprint):s for s in specs}
        for future in concurrent.futures.as_completed(futures):
            s=futures[future]
            try:result=future.result()
            except Exception as e:
                Ledger(OUT/'ledger.jsonl').append(dict(kind='BACKTEST_FAILED',name=s['name'],error=repr(e),fingerprint=fingerprint));raise
            Ledger(OUT/'ledger.jsonl').append(dict(kind='BACKTEST_RESULT',**result,fingerprint=fingerprint));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
