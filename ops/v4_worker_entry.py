"""Bundled offline worker: real frozen weights and a real normalized DEV batch."""
import hashlib,json,platform,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from sidgtn import SIDGTN
ROOT=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def deny(event,args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.system'}:raise PermissionError('offline worker')
sys.addaudithook(deny)

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if manifest['schema_version']!=1 or manifest['command']!=['python','worker_entry.py']:raise ValueError('worker schema')
    for name,h in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=h:raise ValueError('input checksum mismatch')
    model_file=ROOT/'model.pt';saved=torch.load(model_file,map_location='cpu',weights_only=True)
    model=SIDGTN(8,saved['config']['hidden'],saved['config']['dropout']);model.load_state_dict(saved['state_dict']);model.eval()
    batch=np.load(ROOT/'batch.npz',allow_pickle=False)
    price=torch.from_numpy(batch['price']);factors=torch.from_numpy(batch['factors']);asset=torch.from_numpy(batch['asset'])
    begin=time.monotonic()
    with torch.no_grad():cpu=model(price,factors,asset)['quantiles'].numpy()
    max_error=None
    if torch.cuda.is_available():
        model.cuda()
        with torch.no_grad():gpu=model(price.cuda(),factors.cuda(),asset.cuda())['quantiles'].cpu().numpy()
        max_error=float(np.max(np.abs(cpu-gpu)))
        if not np.allclose(cpu,gpu,atol=1e-5,rtol=1e-4):raise ValueError('CPU GPU prediction mismatch')
    out=ROOT/'return';out.mkdir(exist_ok=True);np.save(out/'predictions.npy',cpu)
    shutil.copy2(model_file,out/'model.pt')
    (out/'environment.json').write_text(json.dumps(dict(schema_version=1,machine=platform.node(),python=platform.python_version(),torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),indent=2))
    (out/'metrics.json').write_text(json.dumps(dict(schema_version=1,job_id=manifest['job_id'],status='LOCAL_LOOPBACK_INFERENCE_PASSED',
                batch_rows=len(cpu),cpu_gpu_max_absolute_error=max_error,seconds=time.monotonic()-begin,financial_performance=False),indent=2))
    (out/'training.log').write_text('Protocol smoke uses a frozen model; no remote training claimed.\n')
    result={'schema_version':1,'job_id':manifest['job_id'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'return_manifest.json').write_text(json.dumps(result,indent=2))
    print('LOOPBACK_RETURN_READY')
if __name__=='__main__':main()
