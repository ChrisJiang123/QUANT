"""Real retained archive extraction and multimodal five-horizon dataset, no fits."""
import io,json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.features.registry import compute_factors,PRIMARY
from quant_platform.v5.common import CFG,OUT,read,registered,status
from quant_platform.v5.data import future_open_targets,join_asof
from quant_platform.research.governance import digest_file,write_json,now,offline_guard
from v4_verify_members import Parts

def graph_bars():
    folder=OUT/'data';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'graph_bars.parquet'
    if target.exists():return pd.read_parquet(target)
    verified=read(ROOT/'experiments/v4/archive_verification/metrics.json')
    if not verified['publisher_verified']:raise ValueError('retained archive not verified')
    rows=[];sources=[]
    with zipfile.ZipFile(Parts([ROOT/r['file'] for r in verified['parts']])) as z:
        names=z.namelist()
        for node in CFG['graph_nodes']:
            filename=({'DOGEUSD':'XDGUSD'}.get(node,node))+'_1440.csv'
            matches=[n for n in names if Path(n).name==filename]
            if len(matches)!=1:raise ValueError('fixed graph archive member unavailable: '+node)
            raw=z.read(matches[0]);p=folder/(node+'_1440.csv')
            if not p.exists():p.write_bytes(raw)
            f=pd.read_csv(io.BytesIO(raw),header=None,names=['ts','open','high','low','close','volume','trades'])
            f['candle_ts']=pd.to_datetime(f.ts,unit='s',utc=True);f['pair']=node
            rows.append(f[['pair','candle_ts','open','high','low','close','volume']])
            sources.append(dict(pair=node,member=matches[0],sha256=hashlib.sha256(raw).hexdigest(),rows=len(f),start=str(f.candle_ts.min()),end=str(f.candle_ts.max())))
    b=pd.concat(rows,ignore_index=True).sort_values(['pair','candle_ts'])
    b.to_parquet(target,index=False)
    write_json(folder/'graph_sources.json',dict(nodes=sources,history='each first actual observed trade; fixed named context, not a reconstructed historic universe'))
    return b

def funding():
    folder=OUT/'data';meta=read(ROOT/'experiments/v4/funding_export/whole_metrics.json');rows=[]
    for item in meta['members']:
        if not item['name'].startswith('exports/'):continue
        p=ROOT/item['path']
        if digest_file(p)!=item['sha256']:raise ValueError('retained funding raw changed')
        f=pd.read_csv(p);f['event_at']=pd.to_datetime(f.timestamp,utc=True)
        symbol=f.tradeable.iloc[0];pair='XBTUSD' if 'XBT' in symbol else 'ETHUSD'
        f=f.sort_values('event_at').reset_index(drop=True)
        hours=f.event_at.diff().dt.total_seconds()/3600
        # Raw export settlement cadence switched 4h→1h on 2022-09-29;
        # primary cash research remains scenario-labelled where historic units
        # are inferred. Do not turn gaps into zero or a rate lasting days.
        f['period_hours']=hours.where(hours.isin([1,4]))
        f['interval_end']=f.event_at;f['interval_start']=f.event_at-pd.to_timedelta(f.period_hours,unit='h')
        f['rate']=f.relative_rate;f['pair']=pair;f['source_symbol']=symbol
        f['linear_source']=symbol.startswith('PF_');f['observed_at']=f.event_at
        f['available_at']=f.event_at+pd.Timedelta(hours=6)
        f['received_at']=pd.Timestamp(read(ROOT/'experiments/v4/funding_export/whole_metrics.json').get('collected_at',now()))
        f['history_grade']='CROSS_VENUE_PROXY; historical availability inferred; period from settlement cadence'
        rows.append(f)
    allrows=pd.concat(rows,ignore_index=True)
    allrows.to_parquet(folder/'funding_all.parquet',index=False)
    selected=[];overlap=[]
    for pair,g in allrows.groupby('pair'):
        daily=[]
        for linear,part in g.groupby('linear_source'):
            part=part[part.period_hours.notna()].copy();part['day']=part.event_at.dt.floor('D')
            d=part.groupby('day').agg(rate=('rate','sum'),hours=('period_hours','sum'),n=('rate','size'))
            d=d[d.hours==24];d['pair']=pair;d['linear_source']=linear
            daily.append(d.reset_index())
        merged=pd.concat(daily,ignore_index=True)
        if len(daily)==2:
            a,b=daily;z=a.merge(b,on='day',suffixes=('_a','_b'))
            overlap.append(dict(pair=pair,days=len(z),mean_daily_rate_difference=float((z.rate_a-z.rate_b).mean()),max_abs_difference=float((z.rate_a-z.rate_b).abs().max()),warning='linear/inverse distinct instruments; differences not interchangeable'))
        # Linear preferred only when it exists; inverse earlier rates remain
        # explicitly a different financing proxy, never an inverse price path.
        merged=merged.sort_values(['day','linear_source']).drop_duplicates('day',keep='last')
        merged['available_at']=merged.day+pd.Timedelta(days=1,hours=6)
        selected.append(merged)
    d=pd.concat(selected,ignore_index=True)
    d.to_parquet(folder/'funding_daily.parquet',index=False)
    write_json(folder/'funding_overlap.json',dict(comparisons=overlap,bitmex='previous403; not circumvented',rate_unit_caveat='cadence-inferred historical periods; current hourly rule is not retroactive proof'))
    return d

def build():
    registered();out=OUT/'data';out.mkdir(parents=True,exist_ok=True)
    b=graph_bars();fr=funding()
    # Trade inputs preserve the exact old price snapshot; context nodes use
    # newly extracted archive values, never fill a missing historical candle.
    trade=pd.read_parquet(ROOT/'experiments/v4/bars.parquet')
    cal=pd.date_range('2013-10-06','2026-11-12',freq='D',tz='UTC')
    factors={}
    for p in CFG['graph_nodes']:
        g=(trade[trade.pair==p] if p in CFG['traded_assets'] else b[b.pair==p]).sort_values('candle_ts')
        f=compute_factors(g).assign(pair=p).set_index('candle_ts').reindex(cal)
        factors[p]=f
    graph=np.stack([factors[p][PRIMARY].to_numpy(dtype=np.float32) for p in CFG['graph_nodes']],axis=1)
    graph_mask=np.isfinite(graph).all(axis=2);graph=np.where(graph_mask[:,:,None],graph,0.)
    returns=pd.DataFrame({p:np.log(factors[p].close/factors[p].close.shift()) for p in CFG['graph_nodes']})
    adjacency=np.zeros((len(cal),len(CFG['graph_nodes']),len(CFG['graph_nodes'])),bool)
    for i in range(len(cal)):
        corr=returns.iloc[max(0,i-59):i+1].corr(min_periods=28).to_numpy()
        adjacency[i]=(np.abs(corr)>=.30)|np.eye(len(CFG['graph_nodes']),dtype=bool)
        adjacency[i]&=graph_mask[i,:,None]&graph_mask[i,None,:]
    datasets=[];metadata=[];coverage=[]
    aux_names=['funding_annual','funding_rank30','funding_change7','basis','dvol','fear_greed','wiki_log','event']
    for aid,p in enumerate(CFG['traded_assets']):
        f=factors[p];g=trade[trade.pair==p].sort_values('candle_ts')
        y,v,ym,index=future_open_targets(g,CFG['horizons'])
        y=pd.DataFrame(y,index=index).reindex(cal).to_numpy();v=pd.DataFrame(v,index=index).reindex(cal).to_numpy();ym=np.isfinite(y)&np.isfinite(v)
        available=cal+pd.Timedelta(days=1,hours=6)
        aux=np.zeros((len(cal),len(aux_names)),np.float32);amask=np.zeros_like(aux,dtype=bool)
        rates=fr[fr.pair==p].copy().sort_values('day')
        rates['funding_annual']=rates.rate*365
        rates['funding_rank30']=rates.rate.rolling(30,min_periods=30).rank(pct=True)
        rates['funding_change7']=rates.rate-rates.rate.shift(7)
        vals,m=join_asof(available,rates,['funding_annual','funding_rank30','funding_change7'],max_age_days=3)
        aux[:,:3]=vals;amask[:,:3]=m
        for name,j in [('dvol',4),('fear_greed',5),('wikipedia',6)]:
            path=ROOT/'experiments/v4/prospective'/name/'values.parquet'
            if not path.exists():continue
            x=pd.read_parquet(path).copy()
            # A daily API value at 00:00 can summarize that day's final outcome.
            # Apply conservative +1 full day for DVOL, retain longer existing lags.
            x['available_at']=np.maximum(x.available_at.to_numpy(),(x.observed_at+pd.Timedelta(days=1)).to_numpy())
            if name=='wikipedia':x['value']=np.log1p(x.value)
            vals,m=join_asof(available,x,['value'],max_age_days=7 if name=='wikipedia' else 3)
            aux[:,j]=vals[:,0];amask[:,j]=m[:,0]
        r=np.log(f.close/f.close.shift()).to_numpy();pseq=np.stack([r,((f.high-f.low)/f.close).to_numpy(),np.log1p(f.volume.to_numpy())],axis=1).astype(np.float32)
        x=f[PRIMARY].to_numpy(dtype=np.float32)
        for i in range(CFG['model']['lookback']-1,len(cal)):
            seq=pseq[i-CFG['model']['lookback']+1:i+1]
            if not np.isfinite(seq).all() or not np.isfinite(x[i]).all():continue
            arrays=dict(price=seq,factors=x[i],aux=aux[i],aux_mask=amask[i],nodes=graph[i],node_mask=graph_mask[i],adjacency=adjacency[i],
                sentiment=np.zeros((7,4),np.float32),sentiment_mask=np.zeros(7,bool),asset_id=np.int64(aid),
                target=np.nan_to_num(y[i],nan=0.).astype(np.float32),target_vol=np.nan_to_num(v[i],nan=0.).astype(np.float32),label_mask=ym[i],
                annual_anchor=np.float32(max(CFG['vol_floor'],float(f.vol28.iloc[i]))))
            datasets.append(arrays)
            metadata.append(dict(pair=p,candle_ts=str(cal[i]),available_at=str(available[i]),execution_at=str(cal[i]+pd.Timedelta(days=2)),
                label_end28=str(cal[i]+pd.Timedelta(days=30)),asset_id=aid,evidence='SCENARIO_ONLY',historical_receipt='INFERRED_CLOSE_PLUS_6H'))
        coverage.append(dict(pair=p,aux_nonzero_mask=amask.sum(axis=0).tolist(),primary_valid=int(np.isfinite(x).all(axis=1).sum())))
    packed={k:np.stack([d[k] for d in datasets]) for k in datasets[0]}
    np.savez_compressed(out/'dataset.npz',**packed)
    pd.DataFrame(metadata).to_parquet(out/'samples.parquet',index=False)
    write_json(out/'manifest.json',dict(created_at=now(),dataset_sha256=digest_file(out/'dataset.npz'),samples_sha256=digest_file(out/'samples.parquet'),
        samples=len(metadata),trade_assets=CFG['traded_assets'],node_mapping={p:i for i,p in enumerate(CFG['graph_nodes'])},aux_names=aux_names,
        features=PRIMARY,coverage=coverage,graph_effective_samples=int(graph_mask.sum()),sentiment_real_samples=0,
        blocked_groups=['historical sentiment','historical announced_at calendar','synchronized historical basis','complete U.S. margins/mark path'],
        data_grade='SCENARIO_ONLY; spot prices + cross-venue funding/auxiliary context; not a real U.S. perpetual history'))
    status('V5-DATA','IMPLEMENTED_AND_TESTED',['experiments/v5/data/manifest.json','experiments/v5/us_data/coverage.json'],financial='PARTIAL_COVERAGE_SCENARIO_ONLY')
    status('V5-SENTIMENT','BLOCKED',['experiments/v5/data/manifest.json'],run='REAL_FINANCIAL_EXPERIMENT_NOT_RUN',financial='BLOCKED_DATA',reason='no auditable mapped historical news receipts; integrated masked adapter/model path remains testable')
    print('REAL_DATASET_PREPARED',len(metadata),packed['price'].shape,flush=True)

if __name__=='__main__':
    with offline_guard():build()
