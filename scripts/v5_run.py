"""Unique command logs; shell-free argv; no overwritten failures."""
import os,sys,subprocess,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import digest_file,git_sha,now,write_json,Ledger
def main():
    name=sys.argv[1]; command=sys.argv[2:]
    folder=ROOT/'experiments/v5/commands'/(name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True,exist_ok=False); started=now()
    with (folder/'stdout.log').open('xb') as f:
        p=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'},stdout=f,stderr=subprocess.STDOUT,timeout=14400)
    r=dict(command=command,started_at=started,completed_at=now(),exit_code=p.returncode,git_sha=git_sha(ROOT),log=str((folder/'stdout.log').relative_to(ROOT)),log_sha256=digest_file(folder/'stdout.log'))
    write_json(folder/'receipt.json',r,immutable=True); Ledger(ROOT/'experiments/v5/ledger.jsonl').append(dict(kind='COMMAND',**r))
    print(__import__('json').dumps(r)); print((folder/'stdout.log').read_text(encoding='utf-8',errors='replace')[-3500:]); return p.returncode
if __name__=='__main__':sys.exit(main())
