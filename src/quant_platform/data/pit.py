import json
import numpy as np
import pandas as pd
from quant_platform.research.governance import digest_file, write_json

def pit_join(decisions,values):
    return pd.merge_asof(pd.DataFrame({'decision_at':decisions}).sort_values('decision_at'),
                         values.sort_values('available_at'),left_on='decision_at',right_on='available_at',direction='backward')

def validate_bars(b):
    if b.candle_ts.duplicated().any() or not b.candle_ts.is_monotonic_increasing: raise ValueError('duplicate/unsorted bars')
    if b.candle_ts.dt.tz is None: raise ValueError('UTC required')
    if not np.isfinite(b[['open','high','low','close','volume']]).all().all(): raise ValueError('nonfinite bars')
    if (b[['open','high','low','close']]<=0).any().any() or (b.volume<0).any(): raise ValueError('bad OHLCV')
    if (b.high<b[['open','close','low']].max(axis=1)).any() or (b.low>b[['open','close','high']].min(axis=1)).any(): raise ValueError('bad bounds')
    if (b.available_at<b.candle_ts+pd.Timedelta(days=1)).any(): raise ValueError('future candle available too early')

def split_table(start,end):
    start=pd.Timestamp(start,tz='UTC'); end=pd.Timestamp(end,tz='UTC')
    def segment(a,b,label):
        days=max(0,(b-a).days)
        return dict(start=a.isoformat(),end_exclusive=b.isoformat(),calendar_days=days,independent_28d_windows=days//28,label=label)
    return {'schema_version':1,'boundary_basis':'decision_at UTC; old report final [2025-01-29,2026-09-04)',
            'DEV':segment(start,pd.Timestamp('2025-01-01',tz='UTC'),'DEVELOPMENT'),
            'TOUCHED_AUDIT':segment(pd.Timestamp('2025-01-29',tz='UTC'),pd.Timestamp('2026-09-04',tz='UTC'),'PREVIOUSLY_CONSUMED'),
            'FRESH_HOLDOUT':segment(pd.Timestamp('2026-10-02',tz='UTC'),end-pd.Timedelta(days=28),'CLEAN_BUT_INSUFFICIENT'),
            'FORWARD':{'start':'FREEZE_TIMESTAMP','label':'FORWARD_ONLY'},'purge_days':28,'embargo_days':28}

def load_snapshot(root):
    import pyarrow.parquet as pq
    catalog=json.loads((root/'reports/data_catalog.json').read_text())
    frames=[]; hashes={}; gaps=[]
    for item in catalog:
        m=json.loads((root/item['manifest_path']).read_text())
        for a in m['artifacts']:
            if digest_file(root/a['path'])!=a['sha256']: raise ValueError('TASK001 corruption: '+a['path'])
        hashes[item['parquet_path']]=digest_file(root/item['parquet_path'])
        b=pq.read_table(root/item['parquet_path']).to_pandas()
        f=b.rename(columns={'event_open_ts_utc':'candle_ts','base_volume':'volume'})
        # V4 uses close availability; old artifacts keep their inferred close+60s unchanged.
        f['observed_at']=f['candle_ts']+pd.Timedelta(days=1); f['available_at']=f['observed_at']
        f['processing_version']='v4-close-availability'; f['pair']=item['symbol']
        f['source_quality']='REVIEW_REQUIRED'; f['source_layer']='ARCHIVE'
        f=f[['candle_ts','open','high','low','close','volume','observed_at','available_at','processing_version','pair','source_quality','source_layer']]
        rpath=root/f'experiments/v4/public/kraken_rest_{"btc" if item["symbol"]=="XBTUSD" else "eth"}.raw'
        if rpath.exists():
            raw=json.loads(rpath.read_text())
            if not raw.get('error'):
                result=raw['result']; rows=result[next(k for k in result if k!='last')][:-1]
                r=pd.DataFrame(rows,columns=['ts','open','high','low','close','vwap','volume','count'])
                r['candle_ts']=pd.to_datetime(r.ts,unit='s',utc=True)
                r=r[r.candle_ts>f.candle_ts.max()].copy()
                for c in ['open','high','low','close','volume']: r[c]=pd.to_numeric(r[c])
                r['observed_at']=r.candle_ts+pd.Timedelta(days=1); r['available_at']=r.observed_at
                r['processing_version']='v4-rest-provisional'; r['pair']=item['symbol'];r['source_quality']='REST_PROVISIONAL';r['source_layer']='REST_PROVISIONAL'
                f=pd.concat([f,r[f.columns]],ignore_index=True)
                hashes[str(rpath.relative_to(root))]=digest_file(rpath)
        validate_bars(f)
        missing=pd.date_range(f.candle_ts.min(),f.candle_ts.max(),freq='D').difference(f.candle_ts)
        gaps.extend([dict(pair=item['symbol'],date=str(t),status='REVIEW_REQUIRED',classification='ARCHIVE_INTERVAL_ABSENT_NO_TRADE_PER_PUBLISHER_NOT_INDEPENDENTLY_CONFIRMED') for t in missing])
        frames.append(f)
    result=pd.concat(frames,ignore_index=True)
    return result,dict(schema_version=1,files=hashes,gaps=gaps,quality='REVIEW_REQUIRED',rows=len(result),
                       start=str(result.candle_ts.min()),end=str(result.candle_ts.max()))
