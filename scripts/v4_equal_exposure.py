"""Hindsight average-exposure matched, constant-weight hold comparator, via Freqtrade."""
import json
import pandas as pd
from v4_bootstrap import ROOT
from v4_backtest import backtest,CFG,OUT,REVISION
from v4_engine_metrics import summarize
from quant_platform.research.governance import write_json

def main():
    results=[];basis=[]
    for split in ['DEV','TOUCHED_AUDIT','FRESH_HOLDOUT']:
        source=OUT/'freqtrade_v2'/split/'rule/BASE'
        if not (source/'receipt.json').exists():continue
        metric,_,_=summarize(source);exposure=metric['mean_exposure']
        folder=OUT/'signals'/REVISION/split/'equal_exposure_hold';folder.mkdir(parents=True,exist_ok=True)
        for pair in ['BTC_USD','ETH_USD']:
            f=pd.read_parquet(OUT/'signals'/REVISION/split/'rule'/(pair+'.parquet'))
            f['target_weight']=exposure/2;f['model_version']='equal_exposure_hold'
            f.to_parquet(folder/(pair+'.parquet'),index=False)
        for cost in CFG['cost_per_side']:
            r=backtest('equal_exposure_hold',cost,split);results.append(r)
            if r['exit_code']:raise SystemExit(r['exit_code'])
        basis.append(dict(split=split,source_metrics=str((source/'metrics.json').relative_to(ROOT)),target_total_weight=exposure,
                          retrospective_comparator=True,rebalancing='constant target weight, same fixed cadence; no direction filter'))
    write_json(OUT/'equal_exposure/metrics.json',dict(schema_version=1,basis=basis,commands=results))
if __name__=='__main__':main()
