"""Run an argv without a shell, preserving separate stdout/exit/hash receipts."""
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import digest_file, git_sha, now, write_json

def main():
    name=sys.argv[1]; argv=sys.argv[2:]
    if not argv: raise ValueError('command required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]
    folder=ROOT/'experiments/next_step_20261113/commands'/f'{name}-{stamp}'
    folder.mkdir(parents=True,exist_ok=False)
    started=now()
    with (folder/'run.log').open('xb') as f:
        r=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
    receipt=dict(argv=argv,started_at=started,ended_at=now(),exit_code=r.returncode,
                 git_sha=git_sha(ROOT),log=str((folder/'run.log').relative_to(ROOT)),
                 log_sha256=digest_file(folder/'run.log'))
    write_json(folder/'receipt.json',receipt,immutable=True)
    print(json.dumps(receipt)); print((folder/'run.log').read_text(encoding='utf-8',errors='replace')[-4000:])
    return r.returncode
if __name__=='__main__': sys.exit(main())
