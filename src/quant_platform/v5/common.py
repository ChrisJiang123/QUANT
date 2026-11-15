import json
import sys
from pathlib import Path
from quant_platform.research.governance import Ledger, digest_file, git_sha, now, write_json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/v5'
CFG=json.loads((ROOT/'configs/v5/research.json').read_text(encoding='utf-8'))

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def registered():
    r=read(OUT/'registration.json')
    for p,h in r['files'].items():
        if digest_file(ROOT/p)!=h: raise ValueError('V5 registration changed: '+p)
    return digest_file(OUT/'registration.json')

def status(rid, state, evidence, run='COMPLETED', financial='NOT_EVALUATED', **extra):
    path=OUT/'BUILD_STATE.json'; d=read(path)
    d['requirements'][rid].update(status=state,implementation='IMPLEMENTED' if state.startswith('IMPLEMENTED') else state,
        run=run,financial_evidence=financial,evidence=evidence,updated_at=now(),git_sha=git_sha(ROOT),**extra)
    d['updated_at']=now(); write_json(path,d)
    Ledger(OUT/'ledger.jsonl').append(dict(kind='REQUIREMENT_STATUS',requirement=rid,status=state,evidence=evidence,run=run,financial=financial,**extra))
