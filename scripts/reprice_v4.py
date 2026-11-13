"""Reprice frozen signals with cost-only config changes, without training or final evaluation."""
import argparse,json,sys,zipfile
from pathlib import Path
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from v4_backtest import configuration,OUT
from quant_platform.research.governance import digest,digest_file,ensure_research,write_json,Ledger
from quant_platform.adapters.crypto.offline_freqtrade import run_offline
from v4_engine_metrics import summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('--cost-config',default='configs/v4.json');p.add_argument('--model',default='rule')
    p.add_argument('--split',choices=['DEV','TOUCHED_AUDIT','FRESH_HOLDOUT'],default='DEV');p.add_argument('--scenario',choices=['OPTIMISTIC','BASE','STRESS'],default='BASE');a=p.parse_args()
    cfg=json.loads((ROOT/a.cost_config).read_text());ensure_research(cfg)
    with zipfile.ZipFile(OUT/'frozen_code.zip') as z:original=json.loads(z.read('configs/v4.json'))
    allowed={'cost_per_side','fee_tier'}
    if {k:v for k,v in cfg.items() if k not in allowed}!={k:v for k,v in original.items() if k not in allowed}:raise ValueError('repricing accepts cost/tier changes only')
    fee=float(cfg['cost_per_side'][a.scenario])
    if not 0<fee<.1:raise ValueError('effective fee outside research bounds')
    signal=OUT/'signals/v2'/a.split/a.model
    paths=sorted(signal.glob('*.parquet'))
    if len(paths)!=2:raise ValueError('frozen pair signals missing')
    ident=digest(dict(cost=fee,scenario=a.scenario,model=a.model,split=a.split,signals={f.name:digest_file(f) for f in paths}))
    folder=OUT/'cost_reprice'/ident;folder.mkdir(parents=True,exist_ok=True)
    c=configuration(a.model,signal,folder/'weight_audit.jsonl');write_json(folder/'config.json',c)
    timerange={'DEV':'20161101-20250101','TOUCHED_AUDIT':'20250129-20260904','FRESH_HOLDOUT':'20261002-'}[a.split]
    args=['backtesting','--config',str(folder/'config.json'),'--strategy','V4TargetWeights','--strategy-path',str(ROOT/'freqtrade_userdir/strategies'),
          '--datadir',str(OUT/'freqtrade_v2/data'),'--timerange',timerange,'--fee',str(fee),'--export','trades','--export-filename',str(folder/'result.json'),'--cache','none']
    receipt=folder/'receipt.json'
    if not receipt.exists():
        Ledger(OUT/'ledger.jsonl').append(dict(kind='COST_REPRICE_AUDIT_STARTED',experiment_id=ident,cost=fee,model=a.model,split=a.split))
        result=run_offline(ROOT,args,folder/'run.log');write_json(receipt,result)
        if result['exit_code']:raise SystemExit(result['exit_code'])
    metrics,_,_=summarize(folder)
    write_json(OUT/'cost_reprice/metrics.json',dict(schema_version=1,experiment_id=ident,fee=fee,training_repeated=False,terminal_evaluation_repeated=False,
                 engine_metrics=str((folder/'metrics.json').relative_to(ROOT)),net_return=metrics['net_return'],cache_key_includes_cost_and_signals=True))
    print('FROZEN_SIGNALS_REPRICED_WITHOUT_RETRAINING')
if __name__=='__main__':main()
