"""Independent Qlib expression engine on our own 24/7 UTC calendar and binary data."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/v4/qlib'
def deny_network(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.sendto','socket.gethostbyname'}:raise PermissionError('QLIB_OFFLINE_NO_EXTERNAL_CALLS')
sys.addaudithook(deny_network)

def main():
    OUT.mkdir(exist_ok=True);provider=OUT/'provider'
    bars=pd.read_parquet(ROOT/'experiments/v4/bars.parquet');expected=pd.read_parquet(ROOT/'experiments/v4/features.parquet')
    calendar=pd.date_range(bars.candle_ts.min(),bars.candle_ts.max(),freq='D')
    (provider/'calendars').mkdir(parents=True,exist_ok=True);(provider/'instruments').mkdir(exist_ok=True)
    (provider/'calendars/day.txt').write_text('\n'.join(t.strftime('%Y-%m-%d') for t in calendar)+'\n')
    universe=[]
    for pair,g in bars.groupby('pair'):
        symbol=pair.lower();folder=provider/'features'/symbol;folder.mkdir(parents=True,exist_ok=True)
        f=g.set_index('candle_ts').reindex(calendar)
        for col in ['open','high','low','close','volume']:
            np.concatenate([np.array([0],dtype='<f4'),f[col].to_numpy(dtype='<f4')]).astype('<f4').tofile(folder/(col+'.day.bin'))
        universe.append(f'{pair}\t{g.candle_ts.min():%Y-%m-%d}\t{g.candle_ts.max():%Y-%m-%d}')
    (provider/'instruments/all.txt').write_text('\n'.join(universe)+'\n')
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(provider),region='us',expression_cache=None,dataset_cache=None,kernels=1,joblib_backend='threading')
    expressions=['$close/Ref($close,7)-1','$close/Mean($close,120)-1','Std($close/Ref($close,1)-1,28)*19.1049731745428']
    result=D.features(['XBTUSD','ETHUSD'],expressions,start_time=str(calendar.min().date()),end_time=str(calendar.max().date()),freq='day')
    rows=[]
    for pair in ['XBTUSD','ETHUSD']:
        actual=result.loc[pair].copy();actual.index=pd.to_datetime(actual.index,utc=True)
        f=expected[expected.pair==pair].set_index('candle_ts')
        for name,expression in zip(['mom7','trend120','vol28'],expressions):
            joined=pd.concat([f[name],actual[expression]],axis=1).dropna()
            error=float(np.max(np.abs(joined.iloc[:,0]-joined.iloc[:,1])))
            tolerance=2e-5
            if error>tolerance:raise AssertionError((pair,name,error,tolerance))
            rows.append(dict(pair=pair,factor=name,max_absolute_error=error,tolerance=tolerance,compared_rows=len(joined)))
    result.reset_index().to_parquet(OUT/'expressions.parquet',index=False)
    (OUT/'metrics.json').write_text(json.dumps(dict(schema_version=1,qlib_version=qlib.__version__,status='EXPRESSION_CROSSCHECK_PASSED',
                 calendar='24/7 UTC custom',provider=str(provider),factors=rows,command=sys.argv),indent=2))
    print('QLIB_THREE_FACTOR_CROSSCHECK_PASSED')
if __name__=='__main__':main()
