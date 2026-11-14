"""Fit on DEV, freeze model/code/data, then consume touched and fresh ONCE."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from lightgbm import LGBMRegressor
from quant_platform.models.sidgtn import SIDGTN
from quant_platform.research.governance import *
from quant_platform.research.statistics import forecast_metrics
from quant_platform.features.registry import PRIMARY
from v4_train import dataset,neural_fit,predict_neural,verify_registration,CONFIG,OUT
from v4_bootstrap import ROOT

def fit_frozen():
    verify_registration();folder=OUT/'frozen_model';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():verify_freeze(json.loads((folder/'complete.json').read_text()));print('FROZEN_FIT_RESUME_VERIFIED');return
    m,p,v=dataset();cutoff=pd.Timestamp('2025-01-01',tz='UTC')
    inner_start=cutoff-pd.Timedelta(days=180)
    train=(m.label_available28<cutoff)&m.y28.notna()&np.isfinite(v).all(axis=1)
    inner=train&(m.label_available28<inner_start-pd.Timedelta(days=28));stop=train&(m.decision_at>=inner_start)
    raw=m[PRIMARY].to_numpy(dtype=np.float32);sc=StandardScaler().fit(raw[inner]);x=sc.transform(raw).astype(np.float32)
    ps=StandardScaler().fit(p[inner].reshape(-1,3));p=ps.transform(p.reshape(-1,3)).reshape(p.shape).astype(np.float32)
    y=m[['y14','y28']].to_numpy(dtype=np.float32);aid=m.asset_id.to_numpy()
    joblib.dump({'factor':sc,'price':ps},folder/'normalizers.joblib')
    write_json(folder/'normalizer_audit.json',dict(inner_label_max=str(m.loc[inner,'label_available28'].max()),stop_label_max=str(m.loc[stop,'label_available28'].max()),
                   cutoff=str(cutoff),train_samples=int(inner.sum()),stop_samples=int(stop.sum()),primary=PRIMARY))
    ledger=Ledger(OUT/'ledger.jsonl');env=environment()
    for name in ['ridge','elasticnet','lightgbm']:
        path=folder/(name+'.joblib')
        if path.exists():continue
        ledger.append(dict(kind='TRIAL_STARTED',family=name,scope='final_dev_fit',seed=11,environment=env,git_sha=git_sha(ROOT)))
        models=[]
        for j in range(2):
            model=Ridge(alpha=CONFIG['ridge_alpha']) if name=='ridge' else ElasticNet(alpha=CONFIG['elasticnet_alpha'],l1_ratio=.5,max_iter=3000,random_state=11) if name=='elasticnet' else LGBMRegressor(**CONFIG['lightgbm'],random_state=11,verbosity=-1,n_jobs=4)
            model.fit(x[inner],y[inner,j]);models.append(model)
        joblib.dump(models,path);ledger.append(dict(kind='TRIAL_COMPLETED',family=name,scope='final_dev_fit'))
    stats={}
    for seed in CONFIG['seeds']:
        sf=folder/f'sidgtn_seed{seed}';sf.mkdir(exist_ok=True)
        if (sf/'model.pt').exists() and (sf/'metrics.json').exists():stats[str(seed)]=json.loads((sf/'metrics.json').read_text());continue
        ledger.append(dict(kind='TRIAL_STARTED',family='sidgtn',scope='final_dev_fit',seed=seed,environment=env,git_sha=git_sha(ROOT)))
        model,stat=neural_fit(x,p,aid,y,v,inner.to_numpy(),stop.to_numpy(),CONFIG['sidgtn'],sf,seed)
        write_json(sf/'metrics.json',stat);stats[str(seed)]=stat
        ledger.append(dict(kind='TRIAL_COMPLETED',family='sidgtn',scope='final_dev_fit',seed=seed,**stat))
    write_json(folder/'metrics.json',dict(schema_version=1,training=stats,environment=env))
    write_json(folder/'complete.json',freeze([p for p in folder.rglob('*') if p.is_file()]),immutable=True)
    print('DEV_FIT_COMPLETE_NO_HOLDOUT_EVALUATION')

def freeze_model():
    folder=OUT/'frozen_model';verify_freeze(json.loads((folder/'complete.json').read_text()))
    paths=[p for p in folder.rglob('*') if p.is_file()]
    paths += [OUT/'bars.parquet',OUT/'features.parquet',OUT/'registration.json',ROOT/'configs/v4.json',
              ROOT/'experiments/PREREGISTRATION_V4.md',ROOT/'experiments/SPLITS_V4.json',ROOT/'experiments/HYPOTHESES_V4.md']
    paths += [ROOT/'experiments/MONITORING_RULES_V4.md',ROOT/'experiments/PREREGISTRATION_IMPLEMENTATION_NOTES_V4.md']
    paths += list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'ops').glob('v4_*.py'))+list((ROOT/'freqtrade_userdir/strategies').glob('*.py'))
    manifest=freeze(paths);manifest['git_sha']=git_sha(ROOT)
    write_json(OUT/'frozen.json',manifest,immutable=True)
    write_json(OUT/'freeze_receipt.json',dict(schema_version=1,frozen_at=now(),manifest_sha256=digest_file(OUT/'frozen.json')),immutable=True)
    print('FROZEN',digest_file(OUT/'frozen.json'))

def evaluate(split='final',audit_rerun=False):
    manifest=json.loads((OUT/'frozen.json').read_text());verify_freeze(manifest)
    label='FRESH_HOLDOUT' if split=='final' else 'TOUCHED_AUDIT'
    marker=ROOT/'experiments'/('FINAL_CONSUMED.json' if split=='final' else 'TOUCHED_AUDIT_CONSUMED.json')
    consume_final(marker,manifest,audit_rerun)
    Ledger(OUT/'ledger.jsonl').append(dict(kind='FINAL_EVALUATION_CLAIMED',split=label,manifest_hash=digest(manifest),audit_rerun=audit_rerun))
    m,p,v=dataset();splits=json.loads((ROOT/'experiments/SPLITS_V4.json').read_text());s=splits[label]
    start=pd.Timestamp(s['start']);end=pd.Timestamp(s['end_exclusive'])
    valid=(m.decision_at>=start)&(m.decision_at<end)&m.y28.notna()&np.isfinite(v).all(axis=1)
    subset=m.loc[valid];folder=OUT/'frozen_model';dest=OUT/label;dest.mkdir(exist_ok=True)
    if not valid.any():
        write_json(dest/'metrics.json',dict(schema_version=1,status='INCONCLUSIVE_DATA',samples=0,independent_windows=0));return
    scalers=joblib.load(folder/'normalizers.joblib')
    x=scalers['factor'].transform(m[PRIMARY].to_numpy(dtype=np.float32)[valid]).astype(np.float32)
    p=scalers['price'].transform(p[valid].reshape(-1,3)).reshape((-1,28,3)).astype(np.float32)
    aid=m.asset_id.to_numpy()[valid];preds=[]
    def emit(name,values,o=None):
        for j,h in enumerate([14,28]):
            frame=pd.DataFrame(dict(pair=subset.pair.to_numpy(),decision_at=subset.decision_at.to_numpy(),candle_ts=subset.candle_ts.to_numpy(),
                                     horizon=h,actual=subset[f'y{h}'].to_numpy(),prediction=values[:,j],model=name))
            if o:
                frame['q10']=o['quantiles'][:,j,0];frame['q90']=o['quantiles'][:,j,2];frame['forecast_vol']=o['volatility'][:,j]
            preds.append(frame)
    emit('zero',np.zeros((len(x),2)))
    for name in ['ridge','elasticnet','lightgbm']:
        models=joblib.load(folder/(name+'.joblib'));emit(name,np.stack([model.predict(x) for model in models],axis=-1))
    for seed in CONFIG['seeds']:
        saved=torch.load(folder/f'sidgtn_seed{seed}/model.pt',map_location='cpu',weights_only=False)
        model=SIDGTN(len(PRIMARY),saved['config']['hidden'],saved['config']['dropout']);model.load_state_dict(saved['state_dict']);model.eval()
        o=predict_neural(model,p,x,aid);emit(f'sidgtn_seed{seed}',o['quantiles'][:,:,1],o)
    frame=pd.concat(preds,ignore_index=True);frame.to_parquet(dest/'predictions.parquet',index=False)
    write_json(dest/'metrics.json',dict(schema_version=1,status='INCONCLUSIVE_DATA' if s['independent_28d_windows']<12 else 'AUDIT_ONLY' if label=='TOUCHED_AUDIT' else 'EVALUATED',
             forecasts=forecast_metrics(frame),independent_windows=s['independent_28d_windows'],samples=len(subset),manifest_sha256=digest_file(OUT/'frozen.json')))
    print(label,'EVALUATED_ONCE',len(subset))

def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--fit',action='store_true');g.add_argument('--freeze',action='store_true');g.add_argument('--evaluate',action='store_true')
    parser.add_argument('--split',choices=['final','touched'],default='final');parser.add_argument('--audit-rerun',action='store_true')
    args=parser.parse_args()
    with offline_guard():
        if args.fit:fit_frozen()
        elif args.freeze:freeze_model()
        else:evaluate(args.split,args.audit_rerun)
if __name__=='__main__':main()
