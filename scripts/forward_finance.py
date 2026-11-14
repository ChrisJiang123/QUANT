"""Forward accounting is produced solely by the existing guarded Freqtrade engine."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from quant_platform.research.governance import digest_file, write_json
from quant_platform.research.rule_forward import DAY, POLICY, utc
from v4_backtest import configuration
from v4_engine_metrics import summarize

def engine(config, folder, data_dir, signal_dir, cost, start, end, receipt_time=False):
    folder.mkdir(parents=True, exist_ok=True)
    config.update(v4_data_dir=str(data_dir), v4_signals=str(signal_dir),
                  v4_audit_path=str(folder/'weight_audit.jsonl'))
    write_json(folder/'config.json', config)
    args = ['backtesting','--config',str(folder/'config.json'),'--strategy','V4TargetWeights',
            '--strategy-path',str(ROOT/'freqtrade_userdir'/('receipt_strategies' if receipt_time else 'strategies')),
            '--datadir',str(data_dir),'--timerange',start+'-'+end,'--fee',str(POLICY['cost_per_side'][cost]),
            '--export','trades','--export-filename',str(folder/'result.json'),'--cache','none']
    command = [str(ROOT/'.venv-freqtrade/Scripts/python.exe'),str(ROOT/'ops/receipt_freqtrade_entry.py'),*args]
    env = {**os.environ, 'QUANT_ENGINE_AUDIT': str(folder/'network_audit.jsonl')}
    with (folder/'run.log').open('xb') as f:
        p = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    receipt = dict(command=command, exit_code=p.returncode, log=str(folder/'run.log'),
                   config_hash=digest_file(folder/'config.json'))
    write_json(folder/'receipt.json', receipt)
    if p.returncode: raise RuntimeError('offline Freqtrade failed; see separate run log')
    metrics, curve, fills = summarize(folder)  # Requires actual engine export, not exit code alone.
    return metrics, curve, fills

def simulate(bars, observations, folder):
    """Planned signals are clock-aligned only for engine export, never backdated."""
    superseded={r['record']['supersedes'] for r in observations if 'supersedes' in r['record']}
    eligible = [r['record'] for r in observations if r['record']['eligible_clean'] and r['hash'] not in superseded]
    if not eligible: return dict(status='PENDING_OBSERVATION', performance=None)
    first = min(utc(r['earliest_executable_at']) for r in eligible)
    last = min(bars.groupby('pair').candle_ts.max())
    if last < first+DAY:
        return dict(status='PENDING_EXECUTABLE_CLOSED_BARS', performance=None,
                    initial_research_cash=10000, initial_state='CASH_BEFORE_FIRST_ELIGIBLE_FILL',
                    executed_trades=0, realized_cost=0, current_equity=None,
                    first_planned_open=str(first), latest_closed_candle=str(last),
                    rule=None, buy_hold=None, equal_average_exposure=None)
    data_dir = folder/'data'; data_dir.mkdir(parents=True)
    for pair, g in bars.groupby('pair'):
        name = 'BTC_USD' if pair=='XBTUSD' else 'ETH_USD'
        g.rename(columns={'candle_ts':'date'})[['date','open','high','low','close','volume']].reset_index(drop=True).to_feather(data_dir/(name+'-1d.feather'))
    signals = folder/'signals'
    base = {p: [] for p in ['ETH_USD','BTC_USD']}
    for p in eligible:
        execution = utc(p['earliest_executable_at'])
        for a in p['assets']:
            name = 'BTC_USD' if a['pair']=='XBTUSD' else 'ETH_USD'
            base[name].append(dict(candle_ts=execution-DAY, observed_candle_ts=utc(a['candle_ts']),
                                   target_weight=a['target_weight'], decision=p['rebalance'], gap_mask=False,
                                   available_at=utc(p['received_at']), signal_generated_at=utc(p['generated_at']),
                                   earliest_executable_at=execution))
    metrics = {}; rule_exposure = {}
    for model in ['rule', 'buy_hold', 'equal_exposure_hold']:
        for cost in POLICY['cost_per_side']:
            sig_dir=signals/model/cost; sig_dir.mkdir(parents=True)
            for name, values in base.items():
                f=pd.DataFrame(values).sort_values('candle_ts')
                if model!='rule':
                    exposure=1. if model=='buy_hold' else rule_exposure[cost]
                    f['target_weight']=exposure/2
                    f['decision']=True if model=='equal_exposure_hold' else [True]+[False]*(len(f)-1)
                f.to_parquet(sig_dir/(name+'.parquet'), index=False)
            cfg=configuration(model,sig_dir,folder/'unused.jsonl')
            m, _, _ = engine(cfg,folder/model/cost,data_dir,sig_dir,cost,
                             (first-DAY).strftime('%Y%m%d'),(last+DAY).strftime('%Y%m%d'),True)
            metrics[model+'/'+cost]=m
            if model=='rule': rule_exposure[cost]=m['mean_exposure']
    return dict(status='OFFLINE_ENGINE_RECONSTRUCTION', performance=metrics,
                benchmarks='same first eligible future open; matched exposure retrospective',
                daily_history='append-only per-run snapshots; terminal liquidation is engine convention')
