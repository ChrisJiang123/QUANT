import numpy as np
import pandas as pd

REGISTRY={
 'mom7':('close/lag(close,7)-1',7,'H1'),'mom14':('close/lag(close,14)-1',14,'H1'),
 'mom28':('close/lag(close,28)-1',28,'H1'),'trend120':('close/mean(close,120)-1',120,'H1'),
 'reversal5':('-(close/lag(close,5)-1)',5,'H1'),'reversal14':('-(close/lag(close,14)-1)',14,'H1'),
 'vol7':('std(return,7)*sqrt(365)',8,'H2'),'vol28':('std(return,28)*sqrt(365)',29,'H2'),
 'downside28':('sqrt(mean(min(return,0)^2,28))*sqrt(365)',29,'H2'),
 'atr14':('mean(true_range,14)/close',15,'H2'),'range1':('(high-low)/close',1,'H2'),
 'volume_z28':('(volume-mean(volume,28))/std(volume,28)',28,'H5'),
 'volume_price':('corr(return,volume_change,28)',30,'H5'),
 'illiquidity28':('mean(abs(return)/(volume*close),28)',29,'H5'),
 'relative28':('mom28 - peer_mom28',29,'H1'),'correlation28':('corr(return,peer_return,28)',29,'H2'),
 'beta28':('cov(return,peer_return,28)/var(peer_return,28)',29,'H2'),
}
PRIMARY=['mom7','mom14','mom28','trend120','vol7','vol28','downside28','atr14']

class FactorRegistry:
    def entries(self):
        return [dict(name=k,formula=v[0],lookback=v[1],hypothesis=v[2],primary=k in PRIMARY,
                     source='Kraken OHLCVT',available_at='UTC close; latest dependency',processing_version='v4.1',
                     test='test_truncated_factor_recompute',status='CANDIDATE') for k,v in REGISTRY.items()]

def compute_factors(b):
    # Reindex calendar without filling. Rolling windows that touch absent bars stay missing.
    f=b.set_index('candle_ts').reindex(pd.date_range(b.candle_ts.min(),b.candle_ts.max(),freq='D')).copy()
    f.index.name='candle_ts'; c=f.close; r=c.pct_change(fill_method=None)
    for k in [7,14,28]: f[f'mom{k}']=c/c.shift(k)-1
    f['trend120']=c/c.rolling(120).mean()-1
    for k in [5,14]: f[f'reversal{k}']=-(c/c.shift(k)-1)
    for k in [7,28]: f[f'vol{k}']=r.rolling(k).std()*np.sqrt(365)
    f['downside28']=np.sqrt(r.clip(upper=0).pow(2).rolling(28).mean()*365)
    tr=pd.concat([f.high-f.low,(f.high-c.shift()).abs(),(f.low-c.shift()).abs()],axis=1).max(axis=1)
    f['atr14']=tr.rolling(14).mean()/c
    f['range1']=(f.high-f.low)/c
    f['volume_z28']=(f.volume-f.volume.rolling(28).mean())/f.volume.rolling(28).std().replace(0,np.nan)
    f['volume_price']=r.rolling(28).corr(f.volume.pct_change(fill_method=None))
    f['illiquidity28']=(r.abs()/(f.volume*c).replace(0,np.nan)).rolling(28).mean()
    # Any calendar gap invalidates every primary lookback that depends on it, including momentum endpoints.
    for name in PRIMARY:
        valid=c.notna().rolling(REGISTRY[name][1]+1).sum()==REGISTRY[name][1]+1
        f[name]=f[name].where(valid)
    f['missing_mask']=f[PRIMARY].isna().any(axis=1)
    return f.reset_index()

def quality_report(f):
    x=f[PRIMARY].replace([np.inf,-np.inf],np.nan)
    return dict(schema_version=1,rows=len(f),missing=x.isna().mean().to_dict(),
                extrema={c:dict(min=float(x[c].min()),max=float(x[c].max()),q01=float(x[c].quantile(.01)),q99=float(x[c].quantile(.99))) for c in PRIMARY},
                correlation=x.corr().fillna(0).to_dict(),annual_medians=f.assign(year=f.candle_ts.dt.year).groupby('year')[PRIMARY].median().fillna(0).to_dict(),
                leakage_policy='trailing calendar windows only; truncation tests required; peer features exploratory',
                inference='two-asset cross-sectional IC not used')
