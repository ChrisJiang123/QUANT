"""New forward snapshots are separate from immutable research/terminal snapshots."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from quant_platform.data.pit import validate_bars
from quant_platform.features.registry import compute_factors,PRIMARY
from quant_platform.research.governance import digest_file

def read_forward_bars(path,as_of,require_manifest=True):
    path=Path(path)
    if require_manifest:
        m=json.loads((path.parent/'manifest.json').read_text())
        if m.get('schema_version')!=1 or m.get('files',{}).get(path.name)!=digest_file(path):raise ValueError('forward snapshot manifest/hash mismatch')
        for name,sha in m['files'].items():
            p=(path.parent/name).resolve()
            if not p.is_relative_to(path.parent.resolve()) or digest_file(p)!=sha:raise ValueError('forward source integrity mismatch')
    b=pd.read_parquet(path)
    if set(b.pair.unique())!={'ETHUSD','XBTUSD'}:raise ValueError('fixed spot universe required')
    for pair,g in b.groupby('pair'):validate_bars(g.sort_values('candle_ts'))
    if (b.available_at>pd.Timestamp(as_of)).any():raise ValueError('forward snapshot contains unavailable candles')
    return b

def latest_inputs(b,as_of):
    rows=[];prices=[];assets=[]
    for aid,pair in enumerate(['ETHUSD','XBTUSD']):
        g=b[b.pair==pair].sort_values('candle_ts');f=compute_factors(g)
        f['decision_at']=f.candle_ts+pd.Timedelta(days=1)
        f=f[f.decision_at<=pd.Timestamp(as_of)]
        if f.empty or bool(f.iloc[-1].missing_mask):raise ValueError('latest primary window missing; suppress forward prediction')
        row=f.iloc[-1];r=np.log(f.close/f.close.shift())
        p=np.stack([r.to_numpy(),((f.high-f.low)/f.close).to_numpy(),np.log1p(f.volume.to_numpy())],axis=-1)[-28:]
        if p.shape!=(28,3) or not np.isfinite(p).all():raise ValueError('incomplete forward price window')
        record=row.to_dict();record['pair']=pair;record['asset_id']=aid;rows.append(record);prices.append(p);assets.append(aid)
    latest=pd.DataFrame(rows)
    if (pd.Timestamp(as_of)-latest.decision_at.min())>pd.Timedelta(days=1):raise ValueError('price stale; no forward prediction')
    return latest,np.asarray(prices,dtype=np.float32),np.asarray(assets,dtype=np.int64)

def matured_labels(predictions,bars,as_of):
    labels=[]
    for pred in predictions:
        for h in pred['horizons']:
            maturity=pd.Timestamp(pred['decision_at'])+pd.Timedelta(days=h)
            b=bars[(bars.pair==pred['pair'])&(bars.available_at==maturity)&(bars.available_at<=pd.Timestamp(as_of))]
            if b.empty:continue
            labels.append(dict(pair=pred['pair'],horizon=h,actual_return=float(b.iloc[0].close/pred['anchor_close']-1),available_at=str(maturity)))
    return labels
