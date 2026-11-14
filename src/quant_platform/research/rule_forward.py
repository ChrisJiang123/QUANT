"""Receipt-time, file-only rule observation. No trading client or portfolio engine.

Historical candles are context, not contemporaneous observations. Labels are
executable open-to-open outcomes, conservatively available after the end candle
has closed and its public response has actually arrived.
"""
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.data.pit import validate_bars
from quant_platform.features.registry import PRIMARY, compute_factors
from quant_platform.research.governance import Ledger, digest
from quant_platform.strategy.decision import historical_risk, target_weights

PAIRS = ['ETHUSD', 'XBTUSD']
ORIGIN = pd.Timestamp('2016-11-01', tz='UTC')
DAY = pd.Timedelta(days=1)
POLICY = dict(version='rule-receipt-v1', live_enabled=False, order_enabled=False,
              trend_windows=[28, 60, 120], primary=PRIMARY, target_vol=.15,
              asset_cap=.5, no_trade_band=.05, rebalance_days=14, max_holding_days=28,
              calendar_origin=str(ORIGIN), initial_cash=10000,
              cost_per_side=dict(OPTIMISTIC=.0035, BASE=.0085, STRESS=.012),
              horizons=[14, 28], label='planned-executable-open to open+horizon',
              availability='actual completed response receipt; no clock backdating',
              label_availability='end candle closed and response received',
              correction='append supersedes; corrected observations excluded from clean evidence',
              gaps='no fill; mask affected rolling windows; refuse stale current day',
              execution='strictly future UTC opening; no orders; Freqtrade offline only')

def utc(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None or t.utcoffset().total_seconds() != 0:
        raise ValueError('finite UTC timestamp required')
    return t

def next_open(value):
    return utc(value).floor('D') + DAY

def validate_snapshot(bars, as_of, fresh=True):
    now = utc(as_of)
    if set(bars.pair.unique()) != set(PAIRS):
        raise ValueError('both fixed public spot pairs required')
    if bars[['candle_ts', 'available_at']].isna().any().any():
        raise ValueError('NaT in receipt snapshot')
    for _, g in bars.groupby('pair'):
        g = g.sort_values('candle_ts')
        validate_bars(g)
        for c in ['candle_ts', 'available_at']:
            for t in g[c]: utc(t)
        if (g.candle_ts != g.candle_ts.dt.floor('D')).any():
            raise ValueError('not UTC daily bars')
        if ((g.candle_ts + DAY > g.available_at) | (g.available_at > now)).any():
            raise ValueError('unclosed or unreceived data')
        if fresh and g.candle_ts.max() != now.floor('D') - DAY:
            raise ValueError('latest closed UTC candle missing; suppress prediction')

def latest_rule(bars, as_of):
    validate_snapshot(bars, as_of)
    result = []
    for pair in PAIRS:
        g = bars[bars.pair == pair].sort_values('candle_ts')
        f = compute_factors(g)
        row = f.iloc[-1]
        if bool(row.missing_mask):
            raise ValueError('primary window gap; suppress prediction')
        score = float(np.mean([row.close > f.close.tail(w).mean()
                               for w in POLICY['trend_windows']]))
        result.append(dict(pair=pair, candle_ts=str(row.candle_ts),
                           close=float(row.close), volume_usd=float(row.volume*row.close),
                           score=score, factors={k: float(row[k]) for k in PRIMARY}))
    weights = target_weights(np.array([r['score'] for r in result]),
                             np.array([r['factors']['vol28'] for r in result]))
    if historical_risk(weights, np.array([r['volume_usd'] for r in result]))['status'] == 'REJECT':
        weights = np.zeros(2)
    for r, w in zip(result, weights): r['candidate_weight'] = float(w)
    return result

def historical_signals(features):
    """Causal replacement of the actual V4 rule signal exporter, DEV audit only.

    No next-row missing mask. Missing execution bars must be refused by the
    execution adapter when that bar is reached, rather than predicted in advance.
    """
    f = features.pivot(index='candle_ts', columns='pair', values='close').sort_index()
    times = f.index
    rows = {p: features[features.pair == p].set_index('candle_ts') for p in PAIRS}
    trends = {p: np.mean([f[p] > f[p].rolling(w).mean() for w in POLICY['trend_windows']], axis=0) for p in PAIRS}
    decisions = ((times + DAY - ORIGIN).days % 14 == 0) & (times + DAY >= ORIGIN)
    weights = np.zeros((len(times), 2)); holdings = [None, None]
    for i, t in enumerate(times):
        if not decisions[i]:
            if i: weights[i] = weights[i-1]
            continue
        rr = [rows[p].loc[t] if t in rows[p].index else None for p in PAIRS]
        if any(r is None or bool(r.missing_mask) or not np.isfinite(r.vol28) for r in rr):
            continue
        w = target_weights(np.array([trends[p][i] for p in PAIRS]), np.array([r.vol28 for r in rr]))
        if historical_risk(w, np.array([r.close*r.volume for r in rr]))['status'] == 'REJECT':
            w = np.zeros(2)
        if i and np.max(np.abs(w-weights[i-1])) < .05: w = weights[i-1].copy()
        for j in range(2):
            if holdings[j] is not None and (t-holdings[j]).days >= 28: w[j] = 0.
            if w[j] <= 0: holdings[j] = None
            elif holdings[j] is None: holdings[j] = t
        weights[i] = w
    return {p: pd.DataFrame(dict(candle_ts=times, target_weight=weights[:, j], decision=decisions,
                                gap_mask=f[p].isna().to_numpy(), available_at=times+DAY,
                                earliest_executable_at=times+DAY)) for j, p in enumerate(PAIRS)}

@contextlib.contextmanager
def exclusive(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix+'.transaction-lock')
    for attempt in range(500):
        try:
            with lock.open('x') as f: f.write(str(os.getpid()))
            break
        except FileExistsError:
            if attempt == 499: raise TimeoutError('ledger busy; inspect owner before manual recovery')
            time.sleep(.01)
    try: yield
    finally: lock.unlink()

def append_unique(path, key, payload, content_hash, allow_correction=False):
    """Atomic check+append, even when separate daily processes race."""
    with exclusive(path):
        ledger = Ledger(path); ledger.verify()
        old = [r for r in ledger.rows() if r['record']['key'] == key]
        if old:
            latest = old[-1]
            if latest['record']['content_hash'] == content_hash: return latest, False
            if not allow_correction: raise ValueError('conflicting observation key')
            payload = {**payload, 'classification': 'CORRECTION_BACKFILL',
                       'supersedes': latest['hash'], 'eligible_clean': False}
        return ledger.append({**payload, 'key': key, 'content_hash': content_hash}), True

def make_observation(bars, generated_at, track, run_id, snapshot_hash, previous=None):
    generated = utc(generated_at); rows = latest_rule(bars, generated)
    execution = next_open(generated)
    rebalance = (execution - ORIGIN).days % 14 == 0
    prior = {r['pair']: r for r in previous['assets']} if previous else {}
    for r in rows:
        p = prior.get(r['pair'], {})
        w = r['candidate_weight'] if rebalance else p.get('target_weight', 0.)
        if rebalance and abs(w-p.get('target_weight', 0.)) < .05: w = p.get('target_weight', 0.)
        since = p.get('holding_since')
        if since and execution-utc(since) >= 28*DAY: w = 0.
        r.update(target_weight=float(w), holding_since=(since or str(execution)) if w > 0 else None)
    dependency = []
    for p in PAIRS:
        g = bars[bars.pair == p].sort_values('candle_ts').tail(201)
        dependency += g[['pair','candle_ts','open','high','low','close','volume']].astype(str).to_dict('records')
    receipt = max(bars.available_at)
    return dict(track_id=track, run_id=run_id, generated_at=str(generated), received_at=str(receipt),
                latest_closed_at=str(generated.floor('D')), earliest_executable_at=str(execution),
                classification='FORWARD_LATE_CAPTURE', eligible_clean=True,
                snapshot_hash=snapshot_hash, dependency_hash=digest(dependency),
                rebalance=rebalance, assets=rows, live_enabled=False, order_enabled=False)

def settle(prediction_rows, bars, as_of, path):
    validate_snapshot(bars, as_of, fresh=False)
    as_of = utc(as_of); appended = 0; pending = []
    superseded = {r['record']['supersedes'] for r in prediction_rows if 'supersedes' in r['record']}
    for row in prediction_rows:
        p = row['record']
        if not p['eligible_clean'] or row['hash'] in superseded: continue
        start = utc(p['earliest_executable_at'])
        for h in POLICY['horizons']:
            end = start + h*DAY
            for asset in p['assets']:
                key = f"{row['hash']}:{asset['pair']}:{h}"
                pair = bars[bars.pair == asset['pair']]
                a = pair[pair.candle_ts == start]; b = pair[pair.candle_ts == end]
                if as_of < end or a.empty or b.empty:
                    pending.append(dict(key=key, earliest_maturity=str(end),
                                        earliest_closed_bar_settlement=str(end+DAY)))
                    continue
                value = dict(prediction_hash=row['hash'], horizon=h, pair=asset['pair'],
                             start=str(start), end=str(end), settled_at=str(as_of),
                             available_at=str(max(a.iloc[0].available_at,b.iloc[0].available_at)),
                             actual_return=float(b.iloc[0].open/a.iloc[0].open-1),
                             start_open=float(a.iloc[0].open), end_open=float(b.iloc[0].open))
                # Later revisions of source prices are append-only corrections.
                _, added = append_unique(path, key, value,
                                         digest({k:v for k,v in value.items() if k not in {'settled_at','available_at'}}), True)
                appended += int(added)
    ledger = Ledger(path); ledger.verify()
    latest = {r['record']['key']:r for r in ledger.rows()}
    pending = [p for p in pending if p['key'] not in latest]
    return dict(appended=appended, mature_labels=len(latest), pending_labels=len(pending),
                earliest_maturity=min((p['earliest_maturity'] for p in pending), default=None),
                earliest_closed_bar_settlement=min((p['earliest_closed_bar_settlement'] for p in pending), default=None),
                pending=pending)

def independent_windows(predictions, settlements):
    """Non-overlapping portfolio intervals; two assets never double the count."""
    latest = {r['record']['key']: r['record'] for r in settlements}
    intervals = []
    superseded = {r['record']['supersedes'] for r in predictions if 'supersedes' in r['record']}
    clean = [r for r in predictions if r['record']['eligible_clean'] and r['hash'] not in superseded]
    days = {utc(r['record']['generated_at']).floor('D') for r in clean}
    for r in clean:
        p = r['record']
        if not p['eligible_clean']: continue
        if all(f"{r['hash']}:{pair}:28" in latest for pair in PAIRS):
            start = utc(p['earliest_executable_at'])
            if all(start+(i-1)*DAY in days for i in range(28)):
                intervals.append((start, start+28*DAY))
    count=0; stop=None
    for start,end in sorted(set(intervals)):
        if stop is None or start >= stop: count+=1; stop=end
    return count
