import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from v4_train import dataset,OUT
from quant_platform.features.registry import PRIMARY
from quant_platform.research.observation import monitor,psi
from quant_platform.research.governance import Ledger,write_json,now
from quant_platform.data.forward import read_forward_bars
from quant_platform.features.registry import compute_factors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot');args=parser.parse_args()
    m,_,_=dataset();current=pd.Timestamp(now());last=m.decision_at.max();ref_end=pd.Timestamp('2025-01-01',tz='UTC')-pd.Timedelta(days=208)
    live=m
    if args.snapshot:
        bars=read_forward_bars(ROOT/args.snapshot,current);frames=[]
        for pair,b in bars.groupby('pair'):
            f=compute_factors(b.sort_values('candle_ts'));f['pair']=pair;f['decision_at']=f.candle_ts+pd.Timedelta(days=1);frames.append(f)
        live=pd.concat(frames,ignore_index=True);last=live.decision_at.max()
    values=[]
    for pair,g in m.groupby('pair'):
        reference=g[g.label_available28<ref_end];recent=live[(live.pair==pair)&(live.decision_at>last-pd.Timedelta(days=90))]
        for factor in PRIMARY:values.append(dict(pair=pair,factor=factor,psi=psi(reference[factor].dropna(),recent[factor].dropna())))
    chain=Ledger(OUT/'forward/predictions.jsonl');valid=True
    try:chain.verify()
    except ValueError:valid=False
    known=[r['psi'] for r in values if r['psi'] is not None]
    source_status=[]
    for path in (OUT/'prospective').glob('*/values.parquet'):
        f=pd.read_parquet(path);available=f.loc[f.available_at<=current,'available_at']
        age=(current-available.max()).total_seconds()/86400 if len(available) else None
        source_status.append(dict(source=path.parent.name,age_days=age,masked=age is None or age>3))
    max_age=max([s['age_days'] for s in source_status if s['age_days'] is not None],default=4)
    checks=monitor(source_age_days=max_age,price_present=(current-last)<pd.Timedelta(days=1),chain_valid=valid,
                   psi_value=max(known) if known else None,windows=0)
    write_json(OUT/'monitoring/metrics.json',dict(schema_version=1,checked_at=str(current),price_available_at=str(last),
              checks=checks,feature_psi=values,source_status=source_status,forward_independent_windows=0,performance_status='INCONCLUSIVE_IMMATURE_FORWARD_LABELS',
              no_automatic_execution=True))
    print('OFFLINE_MONITORING_CHECKED')
if __name__=='__main__':main()
