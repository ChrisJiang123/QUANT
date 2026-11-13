"""Daily public-data acquisition followed by a guarded, file-only rule observation."""
import hashlib
import importlib.metadata
import io
import json
import sys
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from quant_platform.research.governance import (Ledger, digest, digest_file, git_sha,
    now, offline_guard, verify_freeze, write_json)
from quant_platform.research.observation import monitor, psi
from quant_platform.research.rule_forward import (DAY, PAIRS, POLICY, append_unique,
    exclusive, independent_windows, make_observation, settle, utc, validate_snapshot)
from forward_finance import simulate

OUT = ROOT/'experiments/rule_forward_v1'
CODE = ['scripts/run_rule_forward.py', 'scripts/forward_finance.py',
        'src/quant_platform/research/rule_forward.py', 'ops/run_forward_daily.ps1',
        'ops/receipt_freqtrade_entry.py', 'freqtrade_userdir/receipt_strategies/V4TargetWeights.py',
        'src/quant_platform/research/governance.py', 'src/quant_platform/features/registry.py',
        'src/quant_platform/strategy/decision.py', 'scripts/v4_engine_metrics.py',
        'scripts/v4_backtest.py', 'freqtrade_userdir/strategies/V4TargetWeights.py',
        'ops/v4_freqtrade_entry.py', 'src/quant_platform/adapters/crypto/offline_freqtrade.py']
CODE += ['scripts/v4_bootstrap.py','src/quant_platform/data/pit.py',
         'src/quant_platform/research/observation.py','src/quant_platform/research/statistics.py',
         'configs/v4.json']

def freeze_track():
    path = OUT/'freeze.json'
    files = {p: digest_file(ROOT/p) for p in CODE}
    environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow']}
    environment['python']=sys.version
    original=ROOT/'experiments/v4/frozen.json'
    verify_freeze(json.loads(original.read_text()))
    original_hash=digest_file(original)
    if path.exists():
        old=json.loads(path.read_text())
        if old['files']!=files or old['policy']!=POLICY:
            raise ValueError('frozen observation code/config changed; a new track is required')
        if old['environment']!=environment or old['legacy_frozen_sha256']!=original_hash:
            raise ValueError('frozen environment or inherited snapshot changed')
        if digest_file(OUT/'code.zip')!=old['code_archive_hash']:
            raise ValueError('frozen code archive damaged')
        return old
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT/'code.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in CODE: z.write(ROOT/p,p)
    core=dict(policy=POLICY, files=files,environment=environment,legacy_frozen_sha256=original_hash)
    value=dict(**core, track_id='rule-receipt-v1-'+digest(core)[:16], frozen_at=now(),
               git_sha=git_sha(ROOT), code_archive_hash=digest_file(OUT/'code.zip'),
               historical_status='exploratory rule; no Alpha promotion; old final remains consumed')
    write_json(path,value,immutable=True)
    return value

def decode_public(raw, pair, received):
    data=json.loads(raw)
    if data.get('error'): raise ValueError('public OHLC source reported error')
    result=data['result']; keys=[k for k in result if k!='last']
    if len(keys)!=1: raise ValueError('unexpected public pair response')
    # Kraken explicitly documents its last entry as uncommitted.
    rows=result[keys[0]][:-1]
    r=pd.DataFrame(rows,columns=['ts','open','high','low','close','vwap','volume','count'])
    r['candle_ts']=pd.to_datetime(r.ts,unit='s',utc=True)
    for c in ['open','high','low','close','volume']: r[c]=pd.to_numeric(r[c])
    r['pair']=pair; r['available_at']=utc(received)
    r=r[r.candle_ts+DAY<=utc(received)]
    if r.empty: raise ValueError('no closed received candles')
    return r[['pair','candle_ts','open','high','low','close','volume','available_at']]

def acquire(folder):
    folder.mkdir(parents=True,exist_ok=False); frames=[]; receipts=[]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for pair in PAIRS:
        url=f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440'
        started=now()
        with opener.open(urllib.request.Request(url,headers={'User-Agent':'QUANT public file research rule-receipt-v1'}),timeout=30) as response:
            raw=response.read(2*1024*1024+1)
            received=now()  # Only after the entire actual response has arrived.
            if len(raw)>2*1024*1024: raise ValueError('public response limit exceeded')
        (folder/(pair+'.json')).write_bytes(raw)
        frames.append(decode_public(raw,pair,received))
        receipts.append(dict(pair=pair,url=url,request_started_at=started,response_received_at=received,
                             raw_sha256=hashlib.sha256(raw).hexdigest(), historical_rows='BACKFILL_CONTEXT_ONLY'))
    bars=pd.concat(frames,ignore_index=True)
    validate_snapshot(bars,now())
    bars.to_parquet(folder/'bars.parquet',index=False)
    write_json(folder/'manifest.json',dict(sources=receipts, bars_sha256=digest_file(folder/'bars.parquet'),
               source_documentation='https://docs.kraken.com/api-reference/market-data/get-ohlc-data',
               quality='REST_PROVISIONAL', available_at_policy=POLICY['availability']),immutable=True)
    return bars

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]
    folder=OUT/'runs'/run_id; folder.mkdir(parents=True,exist_ok=False)
    result=dict(run_id=run_id,started_at=now(),git_sha=git_sha(ROOT),live_enabled=False,order_enabled=False)
    code=1
    try:
        with exclusive(OUT/'daily'):
            frozen=freeze_track()
            predictions=Ledger(OUT/'observations.jsonl'); predictions.verify()
            Ledger(OUT/'settlements.jsonl').verify()
            bars=acquire(folder/'snapshot')  # Explicit public-network phase only.
            with offline_guard():
                generated=now(); previous=predictions.rows()[-1]['record'] if predictions.rows() else None
                record=make_observation(bars,generated,frozen['track_id'],run_id,
                                        digest_file(folder/'snapshot/bars.parquet'),previous)
                key=f"{frozen['track_id']}:{record['latest_closed_at']}"
                row,added=append_unique(OUT/'observations.jsonl',key,record,record['dependency_hash'],True)
                settlements=settle(predictions.rows(),bars,now(),OUT/'settlements.jsonl')
                windows=independent_windows(predictions.rows(),Ledger(OUT/'settlements.jsonl').rows())
                reference=pd.read_parquet(ROOT/'experiments/v4/bars.parquet')
                psis={}
                for pair in PAIRS:
                    old=reference[(reference.pair==pair)&(reference.candle_ts<pd.Timestamp('2025-01-01',tz='UTC'))].sort_values('candle_ts').close.pct_change().dropna()
                    new=bars[bars.pair==pair].sort_values('candle_ts').close.pct_change().dropna().tail(90)
                    psis[pair]=psi(old.to_numpy(),new.to_numpy())
                monitoring=monitor(0,True,True,psi_value=max((v for v in psis.values() if v is not None),default=None),windows=windows)
                finance=simulate(bars,predictions.rows(),folder/'finance')
                existing=Ledger(OUT/'operations.jsonl'); existing.verify()
                day=utc(generated).strftime('%Y-%m-%d')
                days={r['record']['utc_day'] for r in existing.rows() if r['record']['status']=='SUCCESS'}|{day}
                consecutive=0; cursor=utc(generated).floor('D')
                while cursor.strftime('%Y-%m-%d') in days:
                    consecutive+=1; cursor-=DAY
                result.update(status='SUCCESS', completed_at=now(), track_id=frozen['track_id'],
                              snapshot_hash=digest_file(folder/'snapshot/bars.parquet'),
                              generated_at=row['record']['generated_at'], latest_closed_at=record['latest_closed_at'],
                              observation_hash=row['hash'], appended=added, observation_count=predictions.verify(),
                              settlements=settlements, independent_28d_windows=windows,
                              independent_windows_remaining=max(0,12-windows), required_coverage_days=336,
                              operations=dict(actual_unique_utc_days=len(days),consecutive_days=consecutive,
                                              remaining_seven_day_acceptance=max(0,7-consecutive)),
                              monitoring=monitoring, psi=psis, finance=finance)
                existing.append(dict(run_id=run_id,utc_day=day,status='SUCCESS',observation_hash=row['hash'],
                                     appended=added,snapshot_hash=result['snapshot_hash']))
            code=0
    except Exception as exc:
        result.update(status='FAILED_NO_NEW_FINANCIAL_CLAIM', completed_at=now(),
                      error_type=type(exc).__name__, error=str(exc),
                      monitoring=monitor(4,False,'chain' not in str(exc).lower()))
    result['exit_code']=code
    result['code_hashes']={p:digest_file(ROOT/p) for p in CODE}
    write_json(folder/'result.json',result,immutable=True)
    write_json(OUT/'latest.json',dict(run_id=run_id,result=str((folder/'result.json').relative_to(ROOT)),
                                   sha256=digest_file(folder/'result.json')))
    print(json.dumps({k:result[k] for k in ['run_id','status','exit_code']}))
    return code

if __name__=='__main__': sys.exit(run())
