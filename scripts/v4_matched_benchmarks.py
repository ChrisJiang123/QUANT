"""Per-strategy retrospective average-exposure benchmarks, still the same engine."""
import json
from pathlib import Path
import pandas as pd
from v4_bootstrap import ROOT
from v4_backtest import CFG,OUT,configuration
from v4_engine_metrics import summarize
from quant_platform.research.governance import write_json,Ledger
from quant_platform.adapters.crypto.offline_freqtrade import run_offline

def main():
    receipts=[];basis=[]
    for split in ['DEV','TOUCHED_AUDIT','FRESH_HOLDOUT']:
        for source_model in ['ridge','lightgbm','sidgtn_seed11']:
            source=OUT/'freqtrade_v2'/split/source_model/'BASE';metric,_,_=summarize(source);exposure=metric['mean_exposure']
            name='equal_hold_'+source_model;signals=OUT/'signals/v2'/split/name;signals.mkdir(parents=True,exist_ok=True)
            for pair in ['BTC_USD','ETH_USD']:
                f=pd.read_parquet(OUT/'signals/v2'/split/source_model/(pair+'.parquet'))
                f['target_weight']=exposure/2;f['model_version']=name;f.to_parquet(signals/(pair+'.parquet'),index=False)
            for cost,fee in CFG['cost_per_side'].items():
                folder=OUT/'freqtrade_v2'/split/name/cost;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'receipt.json').exists() and json.loads((folder/'receipt.json').read_text())['exit_code']==0:continue
                cfg=configuration('equal_exposure_hold',signals,folder/'weight_audit.jsonl');write_json(folder/'config.json',cfg)
                timerange={'DEV':'20161101-20250101','TOUCHED_AUDIT':'20250129-20260904','FRESH_HOLDOUT':'20261002-'}[split]
                args=['backtesting','--config',str(folder/'config.json'),'--strategy','V4TargetWeights','--strategy-path',str(ROOT/'freqtrade_userdir/strategies'),
                      '--datadir',str(OUT/'freqtrade_v2/data'),'--timerange',timerange,'--fee',str(fee),'--export','trades','--export-filename',str(folder/'result.json'),'--cache','none']
                Ledger(OUT/'ledger.jsonl').append(dict(kind='BACKTEST_STARTED',model=name,cost=cost,split=split,purpose='EXPOSURE_MATCHED_COMPARATOR'))
                result=run_offline(ROOT,args,folder/'run.log');write_json(folder/'receipt.json',result);receipts.append(result)
                Ledger(OUT/'ledger.jsonl').append(dict(kind='BACKTEST_COMPLETED' if result['exit_code']==0 else 'BACKTEST_FAILED',model=name,cost=cost,split=split,**result))
                if result['exit_code']:raise SystemExit(result['exit_code'])
                print(split,name,cost,'COMPLETE',flush=True)
            basis.append(dict(split=split,strategy=source_model,benchmark=name,target_exposure=exposure,retrospective=True,
                              constant_target_weight=True,realized_exposure_may_drift_within_band=True))
    write_json(OUT/'matched_benchmarks/metrics.json',dict(schema_version=1,basis=basis,commands=receipts))
if __name__=='__main__':main()
