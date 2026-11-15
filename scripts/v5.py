"""Versioned offline research CLI. No live, trade or dry-run-bot mode exists."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,CFG,read,registered
from quant_platform.research.governance import digest_file,offline_guard,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dry-check','prepare','train','validate','backtest','observe','replay','resume']);parser.add_argument('--output',default='experiments/v5/manual_replay');parser.add_argument('--start',default='2022-01-01');parser.add_argument('--end',default='2025-01-01');parser.add_argument('--mode',choices=['buy_hold','long_only','asymmetric','filtered','carry','carry_only','tactical'],default='filtered');parser.add_argument('--cost',choices=list(CFG['costs']),default='BASE');args=parser.parse_args()
    registered()
    if args.action in {'dry-check','prepare'}:
        manifest=read(OUT/'data/manifest.json')
        if digest_file(OUT/'data/dataset.npz')!=manifest['dataset_sha256']:raise ValueError('frozen dataset changed')
        for p,h in read(OUT/'v4_preservation.json')['files'].items():
            if digest_file(ROOT/p)!=h:raise ValueError('V4 preservation changed')
        print('V5_DRY_CHECK_OK: registration/data/V4 preservation; no training or financial promotion');return 0
    if args.action in {'replay','resume'}:
        from quant_platform.v5.replay import replay
        import pandas as pd
        output=(ROOT/args.output).resolve()
        if not output.is_relative_to(OUT.resolve()):raise ValueError('research output must stay within experiments/v5')
        if (output/'state/engine.json').exists() and args.action!='resume':raise ValueError('existing state retained; use resume or another output')
        policy=dict(mode=args.mode,cost=args.cost,start=args.start)
        if (output/'policy.json').exists() and read(output/'policy.json')!=policy:raise ValueError('resume policy changed; use another research output')
        if not (output/'policy.json').exists():write_json(output/'policy.json',policy,immutable=True)
        with offline_guard():m=replay(output,mode=args.mode,cost=args.cost,start=args.start,end=args.end,resume=args.action=='resume',prepared=pd.read_pickle(OUT/'replay_inputs.pkl'))
        print(m);return 0
    script={'train':'v5_build_job.py','validate':'v5_validate.py','backtest':'v5_backtest.py','observe':'v5_observe.py'}[args.action]
    return subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT).returncode

if __name__=='__main__':sys.exit(main())
