"""Public single-command replay. Network acquisition and fresh-data consumption excluded."""
import argparse,json,sys
from pathlib import Path
from v4_bootstrap import ROOT,run
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import verify_freeze,digest_file,Ledger

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true',required=True);p.add_argument('--no-external-calls',action='store_true',required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args();out=ROOT/'experiments/v4'
    verify_freeze(json.loads((out/'frozen.json').read_text()));Ledger(out/'ledger.jsonl').verify()
    metrics=json.loads((out/'backtest_summary/metrics.json').read_text())
    for key,item in metrics.items():
        if digest_file(ROOT/item['engine_archive'])!=item['archive_sha256']:raise ValueError('engine artifact corruption: '+key)
    command=[sys.executable,'scripts/run_research_pipeline.py','--offline','--no-external-calls']
    if a.verify_only:command.append('--verify-only')
    if run('replay-primary',command):raise SystemExit(1)
    if not a.verify_only:
        steps=[('replay-scorecard','scripts/v4_scorecard.py'),('replay-extended','scripts/v4_extended_metrics.py'),
               ('replay-delivery','scripts/v4_delivery_audit.py'),('replay-status','scripts/v4_optional_status.py'),('replay-report','scripts/v4_reports.py')]
        for name,script in steps:
            if run(name,[sys.executable,script]):raise SystemExit(1)
    print('FROZEN_SNAPSHOT_OFFLINE_REPLAY_VERIFIED; no terminal or online bot invocation')
if __name__=='__main__':main()
