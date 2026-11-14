"""One-command resume/replay from local immutable snapshots. Never acquires data."""
import argparse,json,sys
from pathlib import Path
from v4_bootstrap import ROOT,run
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import verify_freeze,digest_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true',required=True);p.add_argument('--no-external-calls',action='store_true',required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    frozen=ROOT/'experiments/v4/frozen.json'
    if frozen.exists():verify_freeze(json.loads(frozen.read_text()))
    commands=[('replay-dry-check',[sys.executable,'scripts/v4_train.py','--dry-check','--offline','--no-external-calls'])]
    if not a.verify_only:
        commands.extend([('replay-prepare',[sys.executable,'scripts/v4_prepare.py']),
             ('replay-train',[sys.executable,'scripts/v4_train.py','--train','--offline','--no-external-calls']),
             ('replay-backtests',[sys.executable,'scripts/v4_backtest.py','--offline','--no-external-calls']),
             ('replay-metrics',[sys.executable,'scripts/v4_engine_metrics.py']),
             ('replay-reports',[sys.executable,'scripts/v4_reports.py'])])
    for name,argv in commands:
        code=run(name,argv)
        if code:raise SystemExit(code)
    print('OFFLINE_REPLAY_OK; consumed final sets never silently re-evaluated')
if __name__=='__main__':main()
