"""Explicit online public GET step; append a new snapshot, never rewrite frozen bars."""
import hashlib,json,os,sys,urllib.request
from pathlib import Path
import pandas as pd
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import now,digest_file,write_json,digest
from quant_platform.data.pit import validate_bars

def main():
    out=ROOT/'experiments/v4';base=out/'bars.parquet';frozen=json.loads((out/'frozen.json').read_text())
    if digest_file(base)!=frozen['files'][str(base.resolve())]:raise ValueError('frozen base changed')
    b=pd.read_parquet(base);timestamp=now();frames=[];raws={};sources=[]
    for pair in ['XBTUSD','ETHUSD']:
        url=f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url,headers={'User-Agent':'QuantSIDGTN forward public research/4.4'}),timeout=30) as response:
            raw=response.read(2*1024*1024+1)
            if len(raw)>2*1024*1024:raise ValueError('bounded public response')
        data=json.loads(raw)
        if data.get('error'):raise ValueError('public price source failure; no prediction snapshot')
        result=data['result'];rows=result[next(k for k in result if k!='last')][:-1]
        r=pd.DataFrame(rows,columns=['ts','open','high','low','close','vwap','volume','count'])
        r['candle_ts']=pd.to_datetime(r.ts,unit='s',utc=True)
        for col in ['open','high','low','close','volume']:r[col]=pd.to_numeric(r[col])
        r['observed_at']=r.candle_ts+pd.Timedelta(days=1);r['available_at']=r.observed_at
        r['processing_version']='v4-rest-provisional';r['pair']=pair;r['source_quality']='REST_PROVISIONAL';r['source_layer']='REST_PROVISIONAL'
        prior=b[b.pair==pair];r=r[(r.candle_ts>prior.candle_ts.max())&(r.available_at<=pd.Timestamp(timestamp))]
        merged=pd.concat([prior,r[prior.columns]],ignore_index=True).sort_values('candle_ts');validate_bars(merged);frames.append(merged)
        raws[pair+'_public.json']=raw;sources.append(dict(pair=pair,url=url,collected_at=timestamp,sha256=hashlib.sha256(raw).hexdigest(),new_rows=len(r),quality='REST_PROVISIONAL'))
    snapshot_id=digest(dict(collected_at=timestamp,sources=sources,base_sha256=digest_file(base)))
    folder=ROOT/'data/forward_public'/snapshot_id;folder.mkdir(parents=True,exist_ok=False)
    for name,raw in raws.items():(folder/name).write_bytes(raw);os.chmod(folder/name,0o444)
    pd.concat(frames,ignore_index=True).to_parquet(folder/'bars.parquet',index=False);os.chmod(folder/'bars.parquet',0o444)
    write_json(folder/'manifest.json',dict(schema_version=1,snapshot_id=snapshot_id,collected_at=timestamp,sources=sources,
       base_frozen_sha256=digest_file(base),files={p.name:digest_file(p) for p in folder.iterdir() if p.is_file()},gap_policy='no price fill; incomplete windows masked'),immutable=True)
    target=out/'forward_collection';target.mkdir(exist_ok=True)
    write_json(target/'metrics.json',dict(schema_version=1,snapshot=str((folder/'bars.parquet').relative_to(ROOT)),sources=sources,frozen_data_modified=False))
    print(folder/'bars.parquet')
if __name__=='__main__':main()
