"""Self-contained resumable V5 GPU job. Only NumPy/PyTorch; no network/accounts."""
import argparse,copy,hashlib,json,os,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.model import V5SIDGTN,loss

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def offline(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.sendto'}:raise PermissionError('V5_OFFLINE_WORKER')
sys.addaudithook(offline)

class Arrays(Dataset):
    def __init__(self,arrays,indices):self.arrays=arrays;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return {k:torch.from_numpy(np.asarray(v[self.indices[i]])) for k,v in self.arrays.items()}

def normalize(raw,fit):
    arrays={k:v.copy() for k,v in raw.items()};scalers={}
    for name,maskname in [('price',None),('factors',None),('aux','aux_mask'),('nodes','node_mask'),('sentiment','sentiment_mask')]:
        x=raw[name];flat=x[fit].reshape(-1,x.shape[-1])
        if maskname:
            m=raw[maskname][fit]
            if m.shape==x[fit].shape:
                mask=m.reshape(flat.shape)
            else:mask=np.broadcast_to(m[...,None],x[fit].shape).reshape(flat.shape)
        else:mask=np.isfinite(flat)
        count=mask.sum(axis=0);mean=np.where(mask,flat,0.).sum(axis=0)/np.maximum(count,1)
        var=np.where(mask,(flat-mean)**2,0.).sum(axis=0)/np.maximum(count,1)
        scale=np.sqrt(np.maximum(var,1e-8));z=(x-mean)/scale
        if maskname:
            m=raw[maskname];valid=m if m.shape==x.shape else m[...,None]
            z=np.where(valid,z,0.)
        arrays[name]=z.astype(np.float32)
        scalers[name]=dict(mean=mean.tolist(),scale=scale.tolist(),real_counts=count.tolist())
    return arrays,scalers

def infer(model,arrays,indices,device,batch_size=256):
    out=[];model.eval()
    with torch.no_grad():
        for batch in DataLoader(Arrays(arrays,indices),batch_size=batch_size,shuffle=False,num_workers=0):
            batch={k:v.to(device) for k,v in batch.items()}
            o=model(batch);out.append({k:v.cpu().numpy() for k,v in o.items()})
    return {k:np.concatenate([d[k] for d in out],axis=0) for k in out[0]} if out else {}

def run_one(job,raw,fold,variant,seed,output,device,max_epochs=None):
    folder=output/fold['name']/variant/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'complete.json'
    if complete.exists():
        done=read(complete)
        if done['job_hash']!=job['job_hash']:raise ValueError('resume job mismatch')
        for p,h in done['files'].items():
            if sha(folder/p)!=h:raise ValueError('completed artifact changed')
        print(fold['name'],variant,seed,'RESUME_VERIFIED',flush=True);return
    cfg=job['config'];fit=np.asarray(fold['fit'],dtype=np.int64);stop=np.asarray(fold['stop'],dtype=np.int64)
    test=np.asarray(fold['predict'],dtype=np.int64)
    if len(fit)<100 or len(stop)<30:raise ValueError('insufficient purged training/calibration')
    arrays,scalers=normalize(raw,fit)
    torch.manual_seed(seed);np.random.seed(seed)
    model=V5SIDGTN(hidden=cfg['model']['hidden'],lookback=cfg['model']['lookback'],dropout=cfg['model']['dropout'],variant=variant).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['model']['lr'],weight_decay=cfg['model']['weight_decay'])
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Arrays(arrays,fit),batch_size=cfg['model']['batch_size'],shuffle=True,generator=generator,num_workers=0)
    stop_batch={k:torch.from_numpy(v[stop]).to(device) for k,v in arrays.items()}
    checkpoint=folder/'checkpoint.pt';start=0;best=float('inf');bad=0;best_state=None
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved['job_hash']!=job['job_hash']:raise ValueError('checkpoint source/config mismatch')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;best=saved['best'];bad=saved['bad'];best_state=saved['best_state']
        torch.set_rng_state(saved['rng_cpu'].cpu());generator.set_state(saved['rng_loader'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])
    begin=time.monotonic();gradient_evidence={};initial_parameters={k:v.detach().clone() for k,v in model.named_parameters()}
    if checkpoint.exists():
        gradient_evidence=saved.get('gradient_evidence',{})
        if 'initial_parameters' in saved:initial_parameters={k:v.to(device) for k,v in saved['initial_parameters'].items()}
    epochs=cfg['model']['epochs'] if max_epochs is None else min(max_epochs,cfg['model']['epochs'])
    history=folder/'training.jsonl'
    for epoch in range(start,epochs):
        if bad>=cfg['model']['patience']:break
        model.train();total=0.;samples=0
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
            value,o,terms=loss(model,batch,cfg['loss']);value.backward()
            for prefix in ['price_conv','factor','auxiliary','node_project','edge_attention','sentiment']:
                norm=sum(float(p.grad.detach().abs().sum()) for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None)
                gradient_evidence[prefix]=max(gradient_evidence.get(prefix,0.),norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['model']['gradient_clip']);optimizer.step()
            total+=float(value.detach())*len(batch['price']);samples+=len(batch['price'])
        model.eval()
        with torch.no_grad():value,_,stop_terms=loss(model,stop_batch,cfg['loss']);validation=float(value)
        if validation<best-1e-6:best=validation;bad=0;best_state=copy.deepcopy(model.state_dict())
        else:bad+=1
        with history.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,train_loss=total/samples,purged_calibration_loss=validation,terms=stop_terms,seconds=time.monotonic()-begin))+'\n')
        temp=checkpoint.with_suffix('.tmp')
        torch.save(dict(job_hash=job['job_hash'],epoch=epoch,state=model.state_dict(),optimizer=optimizer.state_dict(),best=best,bad=bad,best_state=best_state,
            gradient_evidence=gradient_evidence,initial_parameters={k:v.cpu() for k,v in initial_parameters.items()},
            rng_cpu=torch.get_rng_state(),rng_loader=generator.get_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None),temp);temp.replace(checkpoint)
        print(fold['name'],variant,seed,'epoch',epoch,round(validation,6),flush=True)
    if best_state is None:raise ValueError('no valid checkpoint')
    model.load_state_dict(best_state);model.eval()
    # Fixed calibration temperature, fitted on inner purged segment only.
    with torch.no_grad():logits=model(stop_batch)['direction_logit'].detach();actual=(stop_batch['target']>0).float();mask=stop_batch['label_mask'].float()
    temp_log=torch.zeros(5,device=device,requires_grad=True);calopt=torch.optim.Adam([temp_log],lr=.05)
    for _ in range(60):
        calopt.zero_grad();temperature=temp_log.exp().clamp(.5,4.)
        value=(torch.nn.functional.binary_cross_entropy_with_logits(logits/temperature,actual,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
        value.backward();calopt.step()
    model.temperature.copy_(temp_log.detach().exp().clamp(.5,4.))
    changed={prefix:sum(float((p.detach()-initial_parameters[k]).abs().sum()) for k,p in model.named_parameters() if k.startswith(prefix)) for prefix in gradient_evidence}
    for prefix in ['price_conv','factor']+(['auxiliary'] if variant!='core' else [])+(['node_project','edge_attention'] if variant in {'graph','sentiment'} else []):
        if gradient_evidence.get(prefix,0.)<=0 or changed.get(prefix,0.)<=0:raise ValueError('branch did not train: '+prefix)
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},constructor=model.constructor,
        variant=variant,seed=seed,job_hash=job['job_hash'],data_hash=job['data_hash'],registration_hash=job['registration_hash'],scalers=scalers,
        node_mapping=job['node_mapping'],horizons=list(HORIZONS_LOCAL)),folder/'model.pt')
    model.to(device);prediction=infer(model,arrays,test,device)
    if prediction:np.savez_compressed(folder/'predictions.npz',indices=test,**prediction)
    batch_ids=np.asarray(job['fixed_batch'],dtype=np.int64)
    fixed=infer(model,arrays,batch_ids,device)
    np.savez_compressed(folder/'fixed_gpu_inference.npz',indices=batch_ids,**fixed)
    write(folder/'metrics.json',dict(variant=variant,seed=seed,fold=fold['name'],device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,parameters=sum(p.numel() for p in model.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,seconds=time.monotonic()-begin,
        best_calibration_loss=best,epochs_completed=epoch+1 if 'epoch' in locals() else start,
        fit_samples=len(fit),calibration_samples=len(stop),predicted_samples=len(test),gradient_norms=gradient_evidence,parameter_change_l1=changed,
        real_aux_samples=int(raw['aux_mask'][fit].any(axis=1).sum()),real_graph_samples=int(raw['node_mask'][fit].sum(axis=1).astype(bool).sum()),
        real_sentiment_samples=int(raw['sentiment_mask'][fit].any(axis=1).sum()),temperature=model.temperature.cpu().tolist()))
    write(folder/'complete.json',dict(job_hash=job['job_hash'],files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}))

HORIZONS_LOCAL=(1,3,7,14,28)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',default='job.json');parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-check',action='store_true');parser.add_argument('--allow-cpu-fixture',action='store_true');parser.add_argument('--max-epochs',type=int)
    args=parser.parse_args();job_path=Path(args.job).resolve();base=job_path.parent;job=read(job_path)
    for p,h in job['files'].items():
        target=(base/p).resolve()
        if not target.is_relative_to(base) or sha(target)!=h:raise ValueError('job file/hash mismatch')
    raw=dict(np.load(base/job['dataset'],allow_pickle=False))
    if sha(base/job['dataset'])!=job['data_hash']:raise ValueError('dataset hash mismatch')
    torch.set_num_threads(4)
    cuda=torch.cuda.is_available();gpu=torch.cuda.get_device_name(0) if cuda else None
    if not args.dry_check and not args.allow_cpu_fixture and (not cuda or '4090' not in gpu):raise RuntimeError('actual RTX4090 required; do not silently use local GPU')
    if args.allow_cpu_fixture and not job.get('fixture_only'):raise PermissionError('CPU override only for isolated fixtures')
    output=base/'results';output.mkdir(exist_ok=True)
    environment=dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,gpu=gpu,
        hostname_hash=hashlib.sha256(__import__('platform').node().encode()).hexdigest(),working_directory_hash=hashlib.sha256(str(base).encode()).hexdigest(),
        current_directory_verified=True,device='cuda' if cuda and not args.allow_cpu_fixture else 'cpu',source_sha=job['source_sha'],data_hash=job['data_hash'])
    write(output/'environment.json',environment)
    if args.dry_check:print('DRY_CHECK_OK',gpu,len(raw['price']));return
    device=environment['device']
    write(output/'job_state.json',dict(state='RUNNING',job_hash=job['job_hash']))
    try:
        for fold in job['folds']:
            for variant in job['variants']:
                for seed in job['seeds']:run_one(job,raw,fold,variant,seed,output,device,args.max_epochs)
        write(output/'job_state.json',dict(state='COMPLETED_NOT_RETURN_VERIFIED',job_hash=job['job_hash']))
    except Exception as exc:
        write(output/'job_state.json',dict(state='FAILED',error=type(exc).__name__+': '+str(exc),job_hash=job['job_hash']));raise
    manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='return_manifest.json'}
    write(output/'return_manifest.json',dict(job_hash=job['job_hash'],source_sha=job['source_sha'],data_hash=job['data_hash'],files=manifest))
    with zipfile.ZipFile(base/'return.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(output))
    print('JOB_COMPLETED_RETURN_ZIP',sha(base/'return.zip'),flush=True)

if __name__=='__main__':main()
