"""Only offline CLI/file boundary; no trading SDK imports in the core package."""
from pathlib import Path
import subprocess

ALLOWED={'backtesting','backtesting-analysis','lookahead-analysis','recursive-analysis','convert-data','list-data'}
FLAGS={'--fee','--config','-c','--strategy','-s','--strategy-path','--datadir','--timerange','--timeframe',
       '--export','--export-filename','--cache','--userdir','--data-format-ohlcv','--pairs','--recursive-analysis-startup-candle',
       '--lookahead-analysis-exportfilename','--minimum-trade-amount','--targeted-trade-amount','--startup-candle','--logfile'}
def authorize_cli(args):
    if not args or args[0] not in ALLOWED: raise PermissionError('Freqtrade subcommand denied')
    if any(';' in str(x) or '\n' in str(x) or (str(x).startswith('-') and x not in FLAGS) for x in args[1:]):
        raise PermissionError('Freqtrade option denied')
    if args[0] in {'backtesting','lookahead-analysis'}:
        if '--fee' not in args: raise PermissionError('explicit fee required')
        fee=float(args[args.index('--fee')+1])
        if not 0<fee<.1: raise PermissionError('invalid cost')

def run_offline(root,args,log):
    authorize_cli(args)
    command=[str(root/'.venv-freqtrade/Scripts/python.exe'),str(root/'ops/v4_freqtrade_entry.py'),*args]
    with Path(log).open('wb') as f:
        p=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    return dict(command=command,exit_code=p.returncode,log=str(Path(log).relative_to(root)))
