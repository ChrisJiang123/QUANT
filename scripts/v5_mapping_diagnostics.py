"""Preregistered raw direction/calibration/median diagnostics, no final tuning."""
import json,sys,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,read,CFG
from quant_platform.v5.replay import replay
from quant_platform.research.governance import write_json,digest_file,digest,offline_guard,Ledger

@offline_guard()
def run(variant,seed,frame,fingerprint):
    folder=OUT/'mapping_diagnostics'/variant/f'seed{seed}'
    if (folder/'complete.json').exists():
        d=read(folder/'complete.json')
        if d['fingerprint']!=fingerprint or digest_file(folder/'metrics.json')!=d['metrics_sha256']:raise ValueError('mapping source changed')
        return read(folder/'metrics.json')
    frame=frame.copy();frame['direction_score']=frame['uncalibrated_score']
    m=replay(folder,mode='model',cost='BASE',predictions=frame,quadrant='mm',prepared=pd.read_pickle(OUT/'replay_inputs.pkl'),variant=variant,seed=seed)
    write_json(folder/'complete.json',dict(fingerprint=fingerprint,metrics_sha256=digest_file(folder/'metrics.json')));return m

def main():
    verified=read(OUT/'gpu_verification.json');returned=ROOT/verified['return_root'];meta=pd.read_parquet(OUT/'data/samples.parquet')
    primary=pd.read_parquet(OUT/'model_predictions.parquet');rows=[];calibration=[]
    for path in returned.glob('wf*/*/seed*/predictions.npz'):
        raw=np.load(path,allow_pickle=False);f=meta.iloc[raw['indices']].reset_index(drop=True).copy()
        prob=1/(1+np.exp(-raw['direction_logit']));f['uncalibrated_score']=(2*prob[:,[3,4]]-1).mean(axis=1)
        f['variant']=path.parent.parent.name;f['seed']=int(path.parent.name[4:]);rows.append(f[['pair','execution_at','variant','seed','uncalibrated_score']])
        g=primary[(primary.variant==f.variant.iloc[0])&(primary.seed==f.seed.iloc[0])&(primary.fold==path.parent.parent.parent.name)].reset_index(drop=True)
        for j,h in enumerate(CFG['horizons']):
            end=pd.to_datetime(g.execution_at,utc=True)+pd.Timedelta(days=h)
            mask=end<pd.Timestamp(str(int(path.parent.parent.parent.name[2:])+1)+'-01-01',tz='UTC')
            truth=(g[f'target_{h}']>0).to_numpy();cal=g[f'direction_probability_{h}'].to_numpy()
            calibration.append(dict(variant=f.variant.iloc[0],seed=int(f.seed.iloc[0]),fold=path.parent.parent.parent.name,horizon=h,
                raw_brier=float(np.mean((prob[mask,j]-truth[mask])**2)),calibrated_brier=float(np.mean((cal[mask]-truth[mask])**2)),
                raw_score_std=float(np.std(2*prob[mask,j]-1)),calibrated_score_std=float(np.std(2*cal[mask]-1))))
    f=primary[primary.fold.str.startswith('wf')].merge(pd.concat(rows,ignore_index=True),on=['pair','execution_at','variant','seed'],validate='one_to_one')
    fingerprint=digest(dict(predictions=digest_file(OUT/'model_predictions.parquet'),source=digest_file(__file__),return_manifest=verified['manifest_sha256']))
    result=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(run,variant,int(seed),g,fingerprint) for (variant,seed),g in f.groupby(['variant','seed'])]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();result.append(r);Ledger(OUT/'ledger.jsonl').append(dict(kind='RAW_MAPPING_DIAGNOSTIC',variant=r['variant'],seed=r['seed'],sharpe=r['sharpe']));print(r['variant'],r['seed'],r['sharpe'],flush=True)
    write_json(OUT/'mapping_diagnostics.json',dict(scope='DEV raw head vs train-only calibrated head vs median; no mapping selection or clean final',calibration=calibration,raw_head_replays=result,fingerprint=fingerprint))

if __name__=='__main__':main()
