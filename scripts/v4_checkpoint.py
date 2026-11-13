import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_platform.research.governance import write_json,now,git_sha
from v4_bootstrap import ROOT

def checkpoint():
    master=(ROOT/'MASTER_PROMPT_FULL_BUILD_V4.4_2026-11-12.md').read_text(encoding='utf-8')
    req={}
    priority=None
    for line in master.splitlines():
        if line.startswith('**P0'):priority='P0'
        elif line.startswith('**P1'):priority='P1'
        elif line.startswith('**P2'):priority='P2'
        if line.startswith('| R-'):
            parts=line.split('|');rid=parts[1].strip()
            req[rid]=dict(priority=priority,description=parts[2].strip(),status='PENDING',evidence=[],command=None,exit_code=None)
    state=ROOT/'experiments/BUILD_STATE.json'
    if state.exists():return
    write_json(state,dict(schema_version=1,updated_at=now(),git_sha=git_sha(ROOT),
                         phase='P0_IMPLEMENTATION_AND_VALIDATION',requirements=req,
                         resume='Read command receipts and per-fold complete.json; never repeat successful folds/final claims.'))
    print('BUILD_STATE initialized',len(req))
if __name__=='__main__':checkpoint()
