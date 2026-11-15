"""Authenticate returned job bytes, all 60 fits and fixed CPU/GPU inference."""
import json,sys,zipfile,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,read,registered,status
from quant_platform.v5.model import V5SIDGTN
from quant_platform.research.governance import digest_file,write_json,now,Ledger

def main():
    registered()
    receipt=read(OUT/'gpu_job.json');base=ROOT/receipt['job_root'];job=read(base/'job.json')
    archive=Path(sys.argv[1]) if len(sys.argv)>1 else base.parent/'return.zip'
    # The manifest binds exact source, dataset and registered job, independently
    # of the transport-level archive hash.
    archive_hash=digest_file(archive)
    target=OUT/'returned'/job['job_id'];target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            p=(target/name).resolve()
            if not p.is_relative_to(target.resolve()):raise ValueError('unsafe return archive path')
            raw=z.read(name)
            if p.exists() and p.read_bytes()!=raw:raise ValueError('returned existing file differs')
            p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():p.write_bytes(raw)
    manifest=read(target/'return_manifest.json')
    for key,value in [('job_hash',job['job_hash']),('source_sha',job['source_sha']),('data_hash',job['data_hash'])]:
        if manifest[key]!=value:raise ValueError('returned identity mismatch: '+key)
    for name,h in manifest['files'].items():
        p=(target/name).resolve()
        if not p.is_relative_to(target.resolve()) or digest_file(p)!=h:raise ValueError('returned hash mismatch '+name)
    if read(target/'job_state.json')['state']!='COMPLETED_NOT_RETURN_VERIFIED':raise ValueError('remote incomplete')
    env=read(target/'environment.json')
    if '4090' not in env['gpu'] or env['device']!='cuda':raise ValueError('not the registered 4090')
    spec=importlib.util.spec_from_file_location('v5_verified_worker',base/'ops/v5_worker_entry.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    torch.set_num_threads(4);raw=dict(np.load(base/'dataset.npz',allow_pickle=False));meta=pd.read_parquet(OUT/'data/samples.parquet')
    checks=[];tables=[]
    for fold in job['folds']:
        for variant in job['variants']:
            for seed in job['seeds']:
                p=target/fold['name']/variant/f'seed{seed}';complete=read(p/'complete.json');m=read(p/'metrics.json')
                if complete['job_hash']!=job['job_hash'] or m['gpu']!=env['gpu']:raise ValueError('fit identity mismatch')
                for name,h in complete['files'].items():
                    if digest_file(p/name)!=h:raise ValueError('fit changed '+str(p/name))
                for prefix in ['price_conv','factor']+(['auxiliary'] if variant!='core' else [])+(['node_project','edge_attention'] if variant=='graph' else []):
                    if m['gradient_norms'][prefix]<=0 or m['parameter_change_l1'][prefix]<=0:raise ValueError('branch not updated')
                saved=torch.load(p/'model.pt',map_location='cpu',weights_only=False)
                if saved['data_hash']!=job['data_hash'] or saved['node_mapping']!=job['node_mapping'] or saved['horizons']!=[1,3,7,14,28]:raise ValueError('checkpoint contract differs')
                model=V5SIDGTN(**saved['constructor']);model.load_state_dict(saved['state_dict']);model.eval()
                arrays,_=worker.normalize(raw,np.array(fold['fit']));ids=np.array(job['fixed_batch'])
                cpu=worker.infer(model,arrays,ids,'cpu');gpu=np.load(p/'fixed_gpu_inference.npz',allow_pickle=False)
                errors={}
                for k,v in cpu.items():
                    errors[k]=float(np.max(np.abs(v.astype(float)-gpu[k].astype(float))))
                    if not np.allclose(v,gpu[k],atol=2e-5,rtol=2e-4):raise ValueError('CPU/GPU mismatch '+k+str(errors[k]))
                checks.append(dict(fold=fold['name'],variant=variant,seed=seed,max_cpu_gpu_absolute_error=max(errors.values()),errors=errors,**{k:m[k] for k in ['epochs_completed','best_calibration_loss','seconds','parameters','peak_cuda_bytes','real_aux_samples','real_graph_samples','real_sentiment_samples']}))
                pred=np.load(p/'predictions.npz',allow_pickle=False);ids=pred['indices'];frame=meta.iloc[ids].reset_index(drop=True).copy()
                for k in ['direction_score','direction_probability','volatility','tail']:
                    for j,h in enumerate([1,3,7,14,28]):frame[f'{k}_{h}']=pred[k][:,j]
                for j,h in enumerate([1,3,7,14,28]):
                    for q,qi in [('q10',0),('q50',1),('q90',2)]:frame[f'{q}_{h}']=pred['quantiles'][:,j,qi]
                    frame[f'target_{h}']=raw['target'][ids,j];frame[f'target_vol_{h}']=raw['target_vol'][ids,j];frame[f'label_mask_{h}']=raw['label_mask'][ids,j]
                frame['annual_anchor']=raw['annual_anchor'][ids];frame['seed']=seed;frame['variant']=variant;frame['fold']=fold['name']
                frame['direction_score']=frame[['direction_score_14','direction_score_28']].mean(axis=1)
                frame['annual_vol']=frame[['volatility_14','volatility_28']].mean(axis=1)
                frame['median_score']=((frame.q50_14/(frame.annual_anchor*np.sqrt(14/365))+frame.q50_28/(frame.annual_anchor*np.sqrt(28/365)))/2).clip(-1,1)
                tables.append(frame)
    pd.concat(tables,ignore_index=True).to_parquet(OUT/'model_predictions.parquet',index=False)
    result=dict(state='VERIFIED',verified_at=now(),archive_sha256=archive_hash,job_hash=job['job_hash'],source_sha=job['source_sha'],data_hash=job['data_hash'],manifest_sha256=digest_file(target/'return_manifest.json'),fits=len(checks),checks=checks,environment=env,atol=2e-5,rtol=2e-4,return_root=str(target.relative_to(ROOT)))
    write_json(OUT/'gpu_verification.json',result);receipt['state']='RETURN_VERIFIED';receipt['return_manifest_sha256']=result['manifest_sha256'];write_json(OUT/'gpu_job.json',receipt)
    for rid in ['V5-GPU','V5-CORE','V5-FACTOR','V5-GRAPH']:status(rid,'IMPLEMENTED_AND_TESTED',['experiments/v5/gpu_verification.json','experiments/v5/model_predictions.parquet'],financial='PENDING_COMPARISON',fits=len(checks))
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','environment']}))

if __name__=='__main__':main()
