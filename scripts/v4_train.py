"""Resumable purged annual walk-forward. All inputs are immutable local snapshots."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from lightgbm import LGBMRegressor
from quant_platform.models.sidgtn import SIDGTN,multitask_loss
from quant_platform.features.registry import PRIMARY
from quant_platform.research.governance import *
from quant_platform.research.statistics import forecast_metrics
from v4_bootstrap import ROOT

OUT=ROOT/'experiments/v4'; CONFIG=json.loads((ROOT/'configs/v4.json').read_text())

def verify_registration():
    r=json.loads((OUT/'registration.json').read_text())
    for p,h in r['files'].items():
        if digest_file(ROOT/p)!=h:raise ValueError('registration changed '+p)
    return r

def dataset():
    f=pd.read_parquet(OUT/'features.parquet');sets=[];prices=[];vols=[]
    for aid,(pair,g) in enumerate(f.groupby('pair',sort=True)):
        g=g.reset_index(drop=True); c=g.close.to_numpy();v=g.volume.to_numpy()
        r=np.log(g.close/g.close.shift()).to_numpy()
        p=np.stack([r,((g.high-g.low)/g.close).to_numpy(),np.log1p(v)],axis=-1)
        ret=pd.Series(r)
        future_vol=np.stack([np.sqrt(sum(ret.shift(-k).pow(2) for k in range(1,h+1))/h*365).to_numpy() for h in [14,28]],axis=-1)
        for i in range(28,len(g)):
            x=g[PRIMARY].iloc[i].to_numpy(dtype=np.float32)
            seq=p[i-27:i+1]
            if np.isfinite(x).all() and np.isfinite(seq).all():
                row=g.iloc[i].to_dict();row['asset_id']=aid;sets.append(row);prices.append(seq);vols.append(future_vol[i])
    m=pd.DataFrame(sets);m['sample_id']=np.arange(len(m))
    return m,np.asarray(prices,dtype=np.float32),np.asarray(vols,dtype=np.float32)

def neural_fit(x,p,aid,y,v,inner,stop,cfg,folder,seed):
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SIDGTN(len(PRIMARY),cfg['hidden'],cfg['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    arrays=[torch.from_numpy(a) for a in [p,x,aid.astype(np.int64),y,v]]
    ds=TensorDataset(*[a[inner] for a in arrays]);loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
    stop_t=[a[stop].to(device) for a in arrays]
    checkpoint=folder/'checkpoint.pt';start_epoch=0;best=float('inf');bad=0;best_state=None
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer'])
        start_epoch=saved['epoch']+1;best=saved['best'];bad=saved['bad'];best_state=saved['best_state']
        torch.set_rng_state(saved['rng_cpu'])
        if device=='cuda' and saved.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(saved['rng_cuda'])
        loader.generator.set_state(saved['rng_loader'])
    begin=time.monotonic();history=folder/'training.jsonl'
    for epoch in range(start_epoch,cfg['epochs']):
        if bad>=cfg['patience']:break
        model.train();total=0.
        for bp,bx,ba,by,bv in loader:
            bp,bx,ba,by,bv=[z.to(device) for z in [bp,bx,ba,by,bv]]
            opt.zero_grad();o=model(bp,bx,ba);loss=multitask_loss(o,by,bv);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(bp)
        model.eval()
        with torch.no_grad():loss=float(multitask_loss(model(stop_t[0],stop_t[1],stop_t[2]),stop_t[3],stop_t[4]))
        if loss<best-1e-6:best=loss;bad=0;best_state=copy.deepcopy(model.state_dict())
        else:bad+=1
        with history.open('a') as log:log.write(json.dumps({'epoch':epoch,'train_loss':total/len(ds),'inner_stop_loss':loss,'seconds':time.monotonic()-begin})+'\n')
        temp=checkpoint.with_suffix('.tmp')
        torch.save(dict(epoch=epoch,state=model.state_dict(),optimizer=opt.state_dict(),best=best,bad=bad,best_state=best_state,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None,rng_loader=loader.generator.get_state()),temp);temp.replace(checkpoint)
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':model.cpu().state_dict(),'config':cfg,'seed':seed},folder/'model.pt')
    model.to(device)
    return model,dict(parameters=sum(a.numel() for a in model.parameters()),seconds=time.monotonic()-begin,device=device,
                      best_inner_loss=best,epochs_completed=epoch+1 if 'epoch' in locals() else start_epoch,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)

def predict_neural(model,p,x,aid):
    device=next(model.parameters()).device;rows=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            o=model(torch.from_numpy(p[start:start+512]).to(device),torch.from_numpy(x[start:start+512]).to(device),torch.from_numpy(aid[start:start+512].astype(np.int64)).to(device))
            rows.append({k:v.cpu().numpy() for k,v in o.items()})
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}

def fit_fold(m,p,v,start,end,label):
    folder=OUT/'training'/label;folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'complete.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved['registration']!=digest_file(OUT/'registration.json'):raise ValueError('fold registration mismatch')
        for file,h in saved['artifacts'].items():
            if digest_file(folder/file)!=h:raise ValueError('fold artifact changed')
        print(label,'RESUME_VERIFIED',flush=True);return
    ledger=Ledger(OUT/'ledger.jsonl');env=environment()
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    cutoff=start-pd.Timedelta(days=28)
    train=(m.label_available28<cutoff)&m.y28.notna()&np.isfinite(v).all(axis=1)
    val=(m.decision_at>=start)&(m.decision_at<end)&m.y28.notna()&np.isfinite(v).all(axis=1)
    # Inner stopping labels mature 28 days before inner validation starts.
    inner_start=cutoff-pd.Timedelta(days=180)
    inner=train&(m.label_available28<inner_start-pd.Timedelta(days=28))
    stop=train&(m.decision_at>=inner_start)
    if inner.sum()<250 or stop.sum()<60 or val.sum()==0:
        write_json(folder/'metrics.json',dict(status='INSUFFICIENT_TRAINING',train=int(inner.sum()),stop=int(stop.sum()),validation=int(val.sum())))
        print(label,'INSUFFICIENT',flush=True);return
    x=m[PRIMARY].to_numpy(dtype=np.float32);sc=StandardScaler().fit(x[inner]);x=sc.transform(x).astype(np.float32)
    ps=StandardScaler().fit(p[inner].reshape(-1,3));p=ps.transform(p.reshape(-1,3)).reshape(p.shape).astype(np.float32)
    y=m[['y14','y28']].to_numpy(dtype=np.float32);aid=m.asset_id.to_numpy()
    write_json(folder/'normalizer.json',dict(factor_mean=sc.mean_.tolist(),factor_scale=sc.scale_.tolist(),price_mean=ps.mean_.tolist(),price_scale=ps.scale_.tolist(),
                fit_max_decision=str(m.loc[inner,'decision_at'].max()),train_label_max=str(m.loc[train,'label_available28'].max()),
                validation_start=str(start),train_cutoff=str(cutoff),train_samples=int(inner.sum()),stop_samples=int(stop.sum()),validation_samples=int(val.sum())))
    predictions=[];sample=m.loc[val]
    def emit(name,pred,extra=None):
        for j,h in enumerate([14,28]):
            frame=pd.DataFrame(dict(sample_id=sample.sample_id.to_numpy(),pair=sample.pair.to_numpy(),decision_at=sample.decision_at.to_numpy(),
                                     candle_ts=sample.candle_ts.to_numpy(),horizon=h,model=name,actual=y[val,j],prediction=pred[:,j],fold=label))
            if extra:
                frame['q10']=extra['quantiles'][:,j,0];frame['q90']=extra['quantiles'][:,j,2]
                frame['direction_probability']=1/(1+np.exp(-extra['direction_logit'][:,j]));frame['forecast_vol']=extra['volatility'][:,j]
                frame['price_gate']=extra['gate'][:,0]
            predictions.append(frame)
    emit('zero',np.zeros((val.sum(),2)))
    for name in ['ridge','elasticnet','lightgbm']:
        begin=time.monotonic();ledger.append(dict(kind='TRIAL_STARTED',family=name,fold=label,seed=11,configuration=CONFIG.get(name,CONFIG.get(name+'_alpha')),git_sha=git_sha(ROOT),data_hash=digest_file(OUT/'features.parquet'),environment=env))
        pred=[];models=[]
        try:
            for j in range(2):
                if name=='ridge':model=Ridge(alpha=CONFIG['ridge_alpha'])
                elif name=='elasticnet':model=ElasticNet(alpha=CONFIG['elasticnet_alpha'],l1_ratio=.5,max_iter=3000,random_state=11)
                else:model=LGBMRegressor(**CONFIG['lightgbm'],random_state=11,verbosity=-1,n_jobs=4)
                model.fit(x[inner],y[inner,j]);pred.append(model.predict(x[val]));models.append(model)
            import joblib
            joblib.dump(models,folder/(name+'.joblib'))
            emit(name,np.stack(pred,axis=-1));ledger.append(dict(kind='TRIAL_COMPLETED',family=name,fold=label,seconds=time.monotonic()-begin))
        except Exception as exc:
            ledger.append(dict(kind='TRIAL_FAILED',family=name,fold=label,error=str(exc)));raise
    seed_metrics={}
    for seed in CONFIG['seeds']:
        name=f'sidgtn_seed{seed}';sf=folder/name;sf.mkdir(exist_ok=True)
        ledger.append(dict(kind='TRIAL_STARTED',family='sidgtn',fold=label,seed=seed,configuration=CONFIG['sidgtn'],environment=env,git_sha=git_sha(ROOT),data_hash=digest_file(OUT/'features.parquet')))
        try:
            model,stats=neural_fit(x,p,aid,y,v,inner.to_numpy(),stop.to_numpy(),CONFIG['sidgtn'],sf,seed)
            begin=time.monotonic();o=predict_neural(model,p[val],x[val],aid[val]);stats['inference_seconds']=time.monotonic()-begin
            emit(name,o['quantiles'][:,:,1],o);seed_metrics[name]=stats
            ledger.append(dict(kind='TRIAL_COMPLETED',family='sidgtn',fold=label,seed=seed,**stats))
            write_json(sf/'metrics.json',stats)
        except Exception as exc:
            ledger.append(dict(kind='TRIAL_FAILED',family='sidgtn',fold=label,seed=seed,error=str(exc)));raise
    result=pd.concat(predictions,ignore_index=True);result.to_parquet(folder/'predictions.parquet',index=False)
    write_json(folder/'metrics.json',dict(forecasts=forecast_metrics(result),training=seed_metrics))
    write_json(complete,dict(registration=digest_file(OUT/'registration.json'),artifacts={str(a.relative_to(folder)):digest_file(a) for a in folder.rglob('*') if a.is_file() and a.name!='complete.json'}))
    print(label,'COMPLETE',len(result),flush=True)

def train():
    verify_registration();m,p,v=dataset();splits=json.loads((ROOT/'experiments/SPLITS_V4.json').read_text())
    starts=[pd.Timestamp('2016-11-01',tz='UTC')]+list(pd.date_range('2018-01-01','2024-01-01',freq='YS',tz='UTC'))
    ends=starts[1:]+[pd.Timestamp(splits['DEV']['end_exclusive'])]
    for i,(start,end) in enumerate(zip(starts,ends)):fit_fold(m,p,v,start,end,f'dev_fold{i}')
    allp=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'training').glob('dev_fold*/predictions.parquet'))],ignore_index=True)
    # Predictions may be used before fold close, but scoring labels must mature
    # within that fold. Keep full prediction files for causal strategy execution.
    fold_ends={f'dev_fold{i}':end for i,end in enumerate(ends)}
    allp['validation_end']=allp['fold'].map(fold_ends)
    allp['metric_eligible']=allp.decision_at+pd.to_timedelta(allp.horizon,unit='D')<allp.validation_end
    allp.to_parquet(OUT/'dev_predictions.parquet',index=False)
    write_json(OUT/'validation/metrics.json',dict(schema_version=1,forecasts=forecast_metrics(allp[allp.metric_eligible]),
                   training_folds=[json.loads(p.read_text()) for p in sorted((OUT/'training').glob('dev_fold*/normalizer.json'))],
                   environment=environment(),registration_sha256=digest_file(OUT/'registration.json')))

def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-check',action='store_true');mode.add_argument('--train',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--split',choices=['validation','final'],default='validation');parser.add_argument('--offline',action='store_true');parser.add_argument('--no-external-calls',action='store_true');parser.add_argument('--audit-rerun',action='store_true')
    args=parser.parse_args();ensure_research(CONFIG)
    with offline_guard():
        verify_registration()
        if args.dry_check:
            m,p,v=dataset();print('DRY_CHECK_OK',len(m),p.shape);return
        if args.train:train()
        elif args.split=='validation':
            p=pd.read_parquet(OUT/'dev_predictions.parquet');write_json(OUT/'validation/metrics.json',dict(schema_version=1,forecasts=forecast_metrics(p[p.metric_eligible]),environment=environment()))
        else:
            from v4_finalize import evaluate
            evaluate('final',args.audit_rerun)
if __name__=='__main__':main()
