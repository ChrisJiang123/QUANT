"""Freeze real purged samples and exact source into a credential-free 4090 job."""
import argparse,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import CFG,OUT,read,registered,status
from quant_platform.research.governance import digest_file,digest,git_sha,now,write_json,Ledger

def fold(meta,start,name,predict=None):
    execution=pd.to_datetime(meta.execution_at,utc=True);end28=pd.to_datetime(meta.label_end28,utc=True)
    start=pd.Timestamp(start);cut=start-pd.Timedelta(days=28);inner=cut-pd.Timedelta(days=180)
    anchor=((execution-pd.Timestamp('2016-11-01',tz='UTC')).dt.days%7==0)
    fit=(end28<inner-pd.Timedelta(days=28))&anchor
    stop=(execution>=inner)&(end28<cut)
    if predict is None:predict=(execution>=start)&(execution<start+pd.DateOffset(years=1))
    result=dict(name=name,fit=np.flatnonzero(fit).tolist(),stop=np.flatnonzero(stop).tolist(),predict=np.flatnonzero(predict).tolist(),
                evaluation_start=str(start),evaluation_end=str(start+pd.DateOffset(years=1)),
                train_label_max=str(end28[fit].max()),calibration_label_max=str(end28[stop].max()),train_cutoff=str(cut),
                scaler_fit_execution_max=str(execution[fit].max()))
    assert end28[fit].max()<inner-pd.Timedelta(days=28)
    assert end28[stop].max()<cut
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    registration=registered();data=OUT/'data';manifest=read(data/'manifest.json')
    if digest_file(data/'dataset.npz')!=manifest['dataset_sha256']:raise ValueError('dataset changed')
    metadata=pd.read_parquet(data/'samples.parquet');raw=np.load(data/'dataset.npz',allow_pickle=False)
    valid=raw['label_mask'].all(axis=1)
    folds=[fold(metadata,f'{year}-01-01T00:00:00Z',f'wf{year}') for year in CFG['walk_forward_years']]
    execution=pd.to_datetime(metadata.execution_at,utc=True)
    folds.append(fold(metadata,'2025-01-01T00:00:00Z','final_dev_fit',execution>=pd.Timestamp('2026-10-01',tz='UTC')))
    for f in folds:
        f['fit']=[i for i in f['fit'] if valid[i]];f['stop']=[i for i in f['stop'] if valid[i]]
    if args.fixture_only:
        folds=folds[:1];folds[0]['fit']=folds[0]['fit'][-140:];folds[0]['stop']=folds[0]['stop'][:40];folds[0]['predict']=folds[0]['predict'][:12]
    source_paths=['ops/v5_worker_entry.py','src/quant_platform/__init__.py','src/quant_platform/v5/__init__.py','src/quant_platform/v5/model.py','configs/v5/research.json']
    source_hash={p:digest_file(ROOT/p) for p in source_paths}
    identity=digest(dict(registration=registration,data_hash=manifest['dataset_sha256'],source=source_hash,folds=folds,fixture=args.fixture_only))[:20]
    folder=OUT/'jobs'/('fixture-' if args.fixture_only else 'gpu-')/identity
    if (folder/'job.json').exists():print(folder);return
    folder.mkdir(parents=True,exist_ok=False)
    for p in source_paths:
        target=folder/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
    shutil.copy2(data/'dataset.npz',folder/'dataset.npz')
    job=dict(job_id=identity,fixture_only=args.fixture_only,created_at=now(),source_sha=git_sha(ROOT),source_hashes=source_hash,
        registration_hash=registration,data_hash=manifest['dataset_sha256'],dataset='dataset.npz',config=CFG,
        variants=CFG['model']['variants'],seeds=[11] if args.fixture_only else CFG['seeds'],folds=folds,
        node_mapping=manifest['node_mapping'],fixed_batch=folds[0]['stop'][:8],
        expected_device='CPU_FIXTURE_ONLY' if args.fixture_only else 'NVIDIA GeForce RTX 4090',
        expected_artifacts=['model.pt','predictions.npz','fixed_gpu_inference.npz','metrics.json','training.jsonl','environment.json','return_manifest.json'],
        source_data_grade=manifest['data_grade'],live_enabled=False,order_enabled=False)
    job['files']={str(p.relative_to(folder)).replace('\\','/'):digest_file(p) for p in folder.rglob('*') if p.is_file()}
    job['job_hash']=digest(job);write_json(folder/'job.json',job,immutable=True)
    launcher='''param([switch]$DryCheck)
$ErrorActionPreference='Stop'
$jobRoot=$PSScriptRoot
$py=@(Get-ChildItem -LiteralPath (Join-Path $env:LOCALAPPDATA 'QuantResearch\\Environments') -Filter python.exe -Recurse | Where-Object { $_.FullName -match '\\\\Scripts\\\\python.exe$' })
if ($py.Count -ne 1) { throw 'Choose the existing verified Python environment; no installation or escalation.' }
$argsList=@('ops\\v5_worker_entry.py','--job','job.json','--resume')
if ($DryCheck) { $argsList += '--dry-check' }
Push-Location -LiteralPath $jobRoot
try { & $py[0].FullName @argsList *> (Join-Path $jobRoot ('worker-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')+'.log')); exit $LASTEXITCODE } finally { Pop-Location }
'''
    (folder/'Run-V5.ps1').write_text(launcher,encoding='utf-8-sig')
    archive=folder.parent/(identity+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob('*'):
            if p.is_file():z.write(p,identity+'/'+p.relative_to(folder).as_posix())
    write_json(OUT/('fixture_job.json' if args.fixture_only else 'gpu_job.json'),dict(job_root=str(folder.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),archive_sha256=digest_file(archive),job_hash=job['job_hash'],state='PREPARED_NOT_STARTED',source_sha=job['source_sha']))
    Ledger(OUT/'ledger.jsonl').append(dict(kind='GPU_JOB_PREPARED' if not args.fixture_only else 'FIXTURE_JOB_PREPARED',job_id=identity,job_hash=job['job_hash'],code_hashes=source_hash,data_hash=job['data_hash']))
    print(folder,archive.stat().st_size)

if __name__=='__main__':main()
