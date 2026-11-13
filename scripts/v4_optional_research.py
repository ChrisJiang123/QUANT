"""DEV-only fixed ablations, no access to terminal predictions/metrics."""
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import v4_train as trainer
from v4_train import dataset,CONFIG,OUT
from v4_bootstrap import ROOT
from v4_backtest import backtest
from quant_platform.models.sidgtn import SIDGTN,multitask_loss
from quant_platform.features.registry import PRIMARY
from quant_platform.research.statistics import forecast_metrics
from quant_platform.research.governance import write_json,Ledger,digest_file,now

def main():
    out=OUT/'optional_research';out.mkdir(exist_ok=True)
    reg=out/'registration.json'
    if not reg.exists():write_json(reg,dict(schema_version=1,registered_at=now(),scope='DEV_ONLY',seed=11,
          ablations=['no_price','no_factor','no_gate','single_task'],hyperparameters=CONFIG['sidgtn'],
          purpose='fixed architecture robustness only; never retest consumed holdout'),immutable=True)
    m,p,v=dataset();xraw=m[PRIMARY].to_numpy(dtype=np.float32);y=m[['y14','y28']].to_numpy(dtype=np.float32);aid=m.asset_id.to_numpy()
    predictions=[];ledger=Ledger(OUT/'ledger.jsonl')
    for mode in ['no_price','no_factor','no_gate','single_task']:
        name=f'sidgtn_ablation_{mode}';parts=[]
        for normalizer in sorted((OUT/'training').glob('dev_fold*/normalizer.json')):
            label=normalizer.parent.name;folder=out/mode/label;folder.mkdir(parents=True,exist_ok=True)
            predfile=folder/'predictions.parquet'
            if predfile.exists():parts.append(pd.read_parquet(predfile));continue
            norm=json.loads(normalizer.read_text());cutoff=pd.Timestamp(norm['train_cutoff']);start=pd.Timestamp(norm['validation_start'])
            ends={f'dev_fold{i}':pd.Timestamp(f'{year}-01-01',tz='UTC') for i,year in enumerate(range(2018,2026))};end=ends[label]
            inner_start=cutoff-pd.Timedelta(days=180)
            train=(m.label_available28<cutoff)&m.y28.notna()&np.isfinite(v).all(axis=1)
            inner=train&(m.label_available28<inner_start-pd.Timedelta(days=28));stop=train&(m.decision_at>=inner_start)
            val=(m.decision_at>=start)&(m.decision_at<end)&m.y28.notna()&np.isfinite(v).all(axis=1)
            x=((xraw-np.array(norm['factor_mean']))/np.array(norm['factor_scale'])).astype(np.float32)
            price=((p-np.array(norm['price_mean']))/np.array(norm['price_scale'])).astype(np.float32)
            trainer.SIDGTN=lambda factors,hidden,dropout:SIDGTN(factors,hidden,dropout,ablation=mode if mode!='single_task' else 'full')
            trainer.multitask_loss=lambda o,r,vol:multitask_loss(o,r,vol,single_task=mode=='single_task')
            ledger.append(dict(kind='TRIAL_STARTED',family='sidgtn',variant=mode,fold=label,seed=11,scope='DEV_ABLATION',registration_sha256=digest_file(reg)))
            model,stats=trainer.neural_fit(x,price,aid,y,v,inner.to_numpy(),stop.to_numpy(),CONFIG['sidgtn'],folder,11)
            o=trainer.predict_neural(model,price[val],x[val],aid[val]);frames=[];sample=m.loc[val]
            for j,h in enumerate([14,28]):
                frames.append(pd.DataFrame(dict(pair=sample.pair.to_numpy(),decision_at=sample.decision_at.to_numpy(),candle_ts=sample.candle_ts.to_numpy(),
                     horizon=h,model=name,actual=y[val,j],prediction=o['quantiles'][:,j,1],q10=o['quantiles'][:,j,0],q90=o['quantiles'][:,j,2],
                     metric_eligible=(sample.decision_at+pd.Timedelta(days=h)<end).to_numpy())))
            frame=pd.concat(frames,ignore_index=True);frame.to_parquet(predfile,index=False);parts.append(frame)
            write_json(folder/'metrics.json',stats);ledger.append(dict(kind='TRIAL_COMPLETED',family='sidgtn',variant=mode,fold=label,seed=11,**stats))
        allp=pd.concat(parts,ignore_index=True);predictions.append(allp)
        for cost in CONFIG['cost_per_side']:
            r=backtest(name,cost,predictions=allp)
            if r['exit_code']:raise SystemExit(r['exit_code'])
        print(mode,'ABLATION_AND_COSTS_COMPLETED',flush=True)
    allp=pd.concat(predictions,ignore_index=True);allp.to_parquet(out/'predictions.parquet',index=False)
    write_json(out/'metrics.json',dict(schema_version=1,status='DEV_ONLY',forecasts=forecast_metrics(allp[allp.metric_eligible]),
                     single_ablation_seed=11,full_model_seeds=CONFIG['seeds'],terminal_reused=False))
if __name__=='__main__':main()
