"""Replay reduced-risk diagnostic controls; no tuned model or cleaner final."""
import sys,json,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,CFG,read
from quant_platform.research.governance import digest,digest_file,write_json,Ledger
from v5_backtest import task

def main():
    primary=read(OUT/'backtest_plan.json');specs=[];details=[]
    for cost in CFG['costs']:
        baselines=[(s['name'],read(OUT/'backtests_v2'/s['name']/'metrics.json')) for s in primary['trials'] if s.get('mode') in ['buy_hold','long_only','asymmetric','filtered','carry'] and s['cost']==cost and not s.get('model_source') and not any(k in s for k in ['margin_hike','favorable_first','funding_stress'])]
        name,base=max(baselines,key=lambda z:z[1]['sharpe'])
        for variant in CFG['model']['variants']:
            models=[read(OUT/'backtests_v2'/f'{variant}_{seed}_mm_{cost}'/'metrics.json') for seed in CFG['seeds']]
            for dimension,key in [('volatility','realized_vol'),('exposure','average_gross')]:
                value=float(np.median([m[key] for m in models]));scale=min(1.,value/max(base[key],1e-9))
                spec=dict(name=f'matched_{variant}_{dimension}_{cost}',mode=base['mode'],cost=cost,exposure_scale=scale)
                specs.append(spec);details.append(dict(name=spec['name'],source_baseline=name,dimension=dimension,model_median_value=value,baseline_value=base[key],scale=scale,
                    interpretation='DEV diagnostic scaling only; never exceeds registered risk cap; integer/band/stops can prevent exact match; no model parameter selected'))
    fingerprint=digest(dict(parent=primary['fingerprint'],script=digest_file(__file__),specs=specs))
    write_json(OUT/'matched_plan.json',dict(fingerprint=fingerprint,comparisons=details,specs=specs))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(task,s,OUT/'replay_inputs.pkl',fingerprint) for s in specs]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();Ledger(OUT/'ledger.jsonl').append(dict(kind='MATCHED_CONTROL',fingerprint=fingerprint,**r));print(json.dumps(r),flush=True)
    for d in details:
        m=read(OUT/'backtests_v2'/d['name']/'metrics.json');key='realized_vol' if d['dimension']=='volatility' else 'average_gross'
        d['replayed_value']=m[key];d['actual_difference_to_model_median']=m[key]-d['model_median_value']
    write_json(OUT/'matched_diagnostics.json',dict(comparisons=details,scope='explicitly in-sample risk/exposure diagnostic; primary common budgets unchanged'))

if __name__=='__main__':main()
