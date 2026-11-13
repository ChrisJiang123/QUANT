import json,sys,shutil,zipfile
from pathlib import Path
import joblib,numpy as np
from v4_bootstrap import ROOT,run
sys.path.insert(0,str(ROOT/'src'))
from v4_train import dataset,OUT
from quant_platform.features.registry import PRIMARY
from quant_platform.workers.jobs import scan_payload,validate_manifest,verify_return
from quant_platform.research.governance import write_json,digest_file,git_sha

def main():
    job=ROOT/'ops/jobs/v4-local-loopback';job.mkdir(parents=True,exist_ok=True)
    files={'worker_entry.py':ROOT/'ops/v4_worker_entry.py','sidgtn.py':ROOT/'src/quant_platform/models/sidgtn.py',
           'model.pt':OUT/'frozen_model/sidgtn_seed11/model.pt'}
    for name,path in files.items():scan_payload(name,path.read_bytes());shutil.copy2(path,job/name)
    m,p,v=dataset();valid=(m.label_available28<__import__('pandas').Timestamp('2025-01-01',tz='UTC'))&m.y28.notna();idx=np.flatnonzero(valid)[-128:]
    scalers=joblib.load(OUT/'frozen_model/normalizers.joblib')
    price=scalers['price'].transform(p[idx].reshape(-1,3)).reshape(-1,28,3).astype(np.float32)
    factors=scalers['factor'].transform(m[PRIMARY].iloc[idx].to_numpy(dtype=np.float32)).astype(np.float32)
    np.savez(job/'batch.npz',price=price,factors=factors,asset=m.asset_id.to_numpy()[idx].astype(np.int64))
    manifest=dict(schema_version=1,job_id='v4-local-loopback',command=['python','worker_entry.py'],git_sha=git_sha(ROOT),
                  files={p.name:digest_file(p) for p in job.iterdir() if p.is_file() and p.name!='manifest.json'},
                  data_snapshot_sha256=digest_file(OUT/'features.parquet'),expected_outputs=['model.pt','predictions.npy','metrics.json','environment.json','training.log','return_manifest.json'])
    validate_manifest(manifest);write_json(job/'manifest.json',manifest)
    bundle=job.with_suffix('.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in job.iterdir():
            if p.is_file():scan_payload(p.name,p.read_bytes());z.write(p,p.name)
    code=run('worker-local-execute',[sys.executable,str(job/'worker_entry.py')])
    if code:raise SystemExit(code)
    returned=json.loads((job/'return/return_manifest.json').read_text());verify_return(job/'return',returned)
    metric=json.loads((job/'return/metrics.json').read_text());metric.update(bundle=str(bundle.relative_to(ROOT)),bundle_sha256=digest_file(bundle),remote_executed=False,return_hashes_verified=True)
    write_json(OUT/'worker/metrics.json',metric)
    print('PACKAGE_EXECUTE_RETURN_HASH_AND_GPU_PARITY_PASSED')
if __name__=='__main__':main()
