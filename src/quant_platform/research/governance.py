"""Research-only guards, hash chained records, immutable freezes and audit receipts."""
import contextlib
import contextvars
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_OFFLINE=contextvars.ContextVar('research_offline',default=False)
def _audit(event,args):
    if _OFFLINE.get() and (event.startswith('socket.') or event in {'http.client.connect','urllib.Request'}):
        raise PermissionError('OFFLINE_NO_EXTERNAL_CALLS')
sys.addaudithook(_audit)

@contextlib.contextmanager
def offline_guard():
    token=_OFFLINE.set(True)
    try: yield
    finally: _OFFLINE.reset(token)

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def digest_file(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def git_sha(root):
    r=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else 'UNVERSIONED'
def write_json(path,value,immutable=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if immutable:
        if path.exists():
            if path.read_text(encoding='utf-8')!=data: raise ValueError('immutable mismatch: '+str(path))
            return
        with path.open('x',encoding='utf-8') as f: f.write(data)
    else:
        temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(data,encoding='utf-8'); temp.replace(path)

class Ledger:
    def __init__(self,path): self.path=Path(path)
    def rows(self):
        if not self.path.exists(): return []
        return [json.loads(s) for s in self.path.read_text(encoding='utf-8').splitlines()]
    def verify(self):
        previous='0'*64
        for row in self.rows():
            sig=row.pop('hash')
            if row['previous_hash']!=previous or digest(row)!=sig: raise ValueError('ledger hash chain broken')
            previous=sig
        return len(self.rows())
    def append(self,record):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        lock=self.path.with_suffix('.lock')
        for attempt in range(500):
            try:lock.open('x').close();break
            except FileExistsError:
                if attempt==499:raise
                time.sleep(.01)
        try:
                self.verify(); rows=self.rows()
                row=dict(schema_version=1,timestamp=now(),previous_hash=rows[-1]['hash'] if rows else '0'*64,record=record)
                row['hash']=digest(row)
                with self.path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        finally: lock.unlink()
        return row

def freeze(paths):
    return {'schema_version':1,'files':{str(Path(p).resolve()):digest_file(p) for p in sorted(paths,key=str)}}
def verify_freeze(manifest):
    for name,sha in manifest['files'].items():
        if not Path(name).is_file() or digest_file(name)!=sha: raise ValueError('frozen file changed: '+name)
def consume_final(marker,manifest,audit_rerun=False):
    verify_freeze(manifest); marker=Path(marker)
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['manifest_hash']!=digest(manifest): raise PermissionError('FINAL_MANIFEST_CHANGED')
        if not audit_rerun: raise PermissionError('FINAL_ALREADY_CONSUMED')
        return
    write_json(marker,{'schema_version':1,'consumed_at':now(),'manifest_hash':digest(manifest)},immutable=True)

def environment():
    import torch
    return dict(machine=platform.node(),system=platform.platform(),python=sys.version,torch=torch.__version__,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def ensure_research(config):
    if config.get('live_enabled') is not False or config.get('order_enabled') is not False:
        raise PermissionError('RESEARCH_ONLY_REQUIRED')
    def scan(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if any(t in k.lower() for t in ('api_key','apikey','secret','password','token')): raise PermissionError('credential field forbidden')
                scan(x)
        elif isinstance(v,list):
            for x in v: scan(x)
    scan(config)
