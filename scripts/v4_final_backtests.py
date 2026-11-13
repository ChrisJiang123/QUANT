import json,sys
from pathlib import Path
import pandas as pd
from v4_bootstrap import ROOT
from v4_backtest import backtest,CFG,OUT
from quant_platform.research.governance import write_json

def main():
    results=[]
    models=['buy_hold','sma200','rule','direction_only','ridge','elasticnet','lightgbm']+[f'sidgtn_seed{s}' for s in CFG['seeds']]
    for split in ['TOUCHED_AUDIT','FRESH_HOLDOUT']:
        p=OUT/split/'predictions.parquet'
        if not p.exists():continue
        predictions=pd.read_parquet(p)
        for model in models:
            for cost in CFG['cost_per_side']:
                r=backtest(model,cost,split,predictions);results.append(r)
                if r['exit_code']:raise SystemExit(r['exit_code'])
    write_json(OUT/'final_backtests_runs.json',results)
if __name__=='__main__':main()
