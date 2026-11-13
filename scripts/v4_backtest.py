"""Generate PIT signals, convert validated candles and run the sole Freqtrade engine."""
import argparse
import json
import sys
import os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from quant_platform.strategy.decision import target_weights,historical_risk
from quant_platform.adapters.crypto.offline_freqtrade import run_offline
from quant_platform.research.governance import write_json,digest_file,offline_guard,Ledger,git_sha
from v4_bootstrap import ROOT

OUT=ROOT/'experiments/v4';CFG=json.loads((ROOT/'configs/v4.json').read_text())
REVISION=os.environ.get('V4_BT_REVISION','v2')
BT=OUT/('freqtrade' if REVISION=='v1' else 'freqtrade_v2')
ORIGIN=pd.Timestamp('2016-11-01',tz='UTC')

def export_data():
    bars=pd.read_parquet(OUT/'bars.parquet');directory=BT/'data';directory.mkdir(parents=True,exist_ok=True)
    receipt=[]
    for pair,g in bars.groupby('pair'):
        f=g.rename(columns={'candle_ts':'date'})[['date','open','high','low','close','volume']].reset_index(drop=True)
        name='BTC_USD' if pair=='XBTUSD' else 'ETH_USD';path=directory/(name+'-1d.feather');f.to_feather(path)
        actual=pd.read_feather(path)
        pd.testing.assert_frame_equal(f,actual)
        receipt.append(dict(pair=pair,rows=len(f),path=str(path.relative_to(ROOT)),sha256=digest_file(path),identical=True))
    write_json(BT/'conversion/metrics.json',dict(schema_version=1,files=receipt))

def signals(model,predictions=None,split='DEV'):
    features=pd.read_parquet(OUT/'features.parquet');folder=OUT/'signals'/REVISION/split/model;folder.mkdir(parents=True,exist_ok=True)
    if all((folder/(p+'.parquet')).exists() for p in ['BTC_USD','ETH_USD']):return folder
    f=features.pivot(index='candle_ts',columns='pair',values='close').sort_index()
    times=f.index;weights=np.zeros((len(times),2));decision=((times+pd.Timedelta(days=1)-ORIGIN).days%14==0)&(times+pd.Timedelta(days=1)>=ORIGIN)
    if predictions is not None:
        pm=predictions[(predictions.model==model)&(predictions.horizon==28)].pivot(index='candle_ts',columns='pair',values='prediction')
        pred=pm.reindex(index=times,columns=['ETHUSD','XBTUSD']).to_numpy()
    holdings=[None,None]
    for i,t in enumerate(times):
        if not decision[i]:
            if i:weights[i]=weights[i-1]
            continue
        rows=[]
        for pair in ['ETHUSD','XBTUSD']:
            row=features[(features.pair==pair)&(features.candle_ts==t)]
            rows.append(row.iloc[0] if len(row) else None)
        if any(r is None or not np.isfinite(r.vol28) or bool(r.missing_mask) for r in rows):continue
        if model=='buy_hold':w=np.array([.5,.5])
        elif model=='equal_exposure_hold':raise ValueError('requires exposure supplied from frozen strategy')
        elif model=='sma200':w=np.array([.5 if t in f.index and len(f.loc[:t])>=200 and f.loc[t,pair]>f.loc[:t,pair].iloc[-200:].mean() else 0 for pair in ['ETHUSD','XBTUSD']])
        else:
            score=np.array([np.mean([r.close/features[(features.pair==r.pair)&(features.candle_ts<=t)].close.tail(window).mean()>1 for window in CFG['trend_windows']]) for r in rows])
            if model not in {'rule','direction_only'}:score=np.clip(.5+pred[i]/.1,0,1)
            if not np.isfinite(score).all():continue
            w=np.minimum(.5,score*.5) if model=='direction_only' else target_weights(score,np.array([r.vol28 for r in rows]),source='rule' if model=='rule' else model.split('_')[0])
        volumes=np.array([r.volume*r.close for r in rows])
        risk=historical_risk(w,volumes)
        if risk['status']=='REJECT':w=np.zeros(2)
        if i and np.max(np.abs(w-weights[i-1]))<CFG['no_trade_band']:w=weights[i-1]
        if model not in {'buy_hold','equal_exposure_hold'}:
            for j in range(2):
                if holdings[j] is not None and (t-holdings[j]).days>=28:w[j]=0.
                if w[j]<=0:holdings[j]=None
                elif holdings[j] is None:holdings[j]=t
        weights[i]=w
    for j,pair in enumerate(['ETHUSD','XBTUSD']):
        missing=f[pair].isna()|f[pair].shift(-1).isna()
        s=pd.DataFrame(dict(schema_version=1,candle_ts=times,target_weight=weights[:,j],decision=decision,gap_mask=missing.to_numpy(),
                            model_version=model,available_at=times+pd.Timedelta(days=1),earliest_executable_at=times+pd.Timedelta(days=1),
                            venue='kraken',pair=pair,quote_currency='USD',processing_version='v4.1',experiment_id='v4-'+split,
                            signal_generated_at=times+pd.Timedelta(days=1)))
        s.to_parquet(folder/(('BTC_USD' if pair=='XBTUSD' else 'ETH_USD')+'.parquet'),index=False)
    return folder

def configuration(model,signal_folder,audit_path):
    return dict(schema_version=1,live_enabled=False,order_enabled=False,dry_run=True,
       max_open_trades=2,stake_currency='USD',stake_amount='unlimited',tradable_balance_ratio=.99,
       dry_run_wallet=10000,timeframe='1d',trading_mode='spot',margin_mode='',
       entry_pricing={'price_side':'other','use_order_book':False,'order_book_top':1,'price_last_balance':0.0,'check_depth_of_market':{'enabled':False,'bids_to_ask_delta':1}},
       exit_pricing={'price_side':'other','use_order_book':False,'order_book_top':1},
       exchange={'name':'kraken','pair_whitelist':['BTC/USD','ETH/USD'],'pair_blacklist':[],
                 'enable_ws':False,'ccxt_config':{'enableRateLimit':False}},
       pairlists=[{'method':'StaticPairList'}],unfilledtimeout={'entry':10,'exit':10,'unit':'minutes'},
       dataformat_ohlcv='feather',dataformat_trades='feather',
       v4_signals=str(signal_folder),v4_audit_path=str(audit_path),v4_max_holding=model not in {'buy_hold','equal_exposure_hold'},
       v4_rebalance=model!='buy_hold',
       v4_data_dir=str(BT/'data'),
       strategy='V4TargetWeights',user_data_dir=str(ROOT/'freqtrade_userdir'))

def backtest(model,cost,split='DEV',predictions=None):
    folder=BT/split/model/cost;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['exit_code']==0:return r
    signal_folder=signals(model,predictions,split)
    config=configuration(model,signal_folder,folder/'weight_audit.jsonl');write_json(folder/'config.json',config)
    timerange={'DEV':'20161101-20250101','TOUCHED_AUDIT':'20250129-20260904','FRESH_HOLDOUT':'20261002-'}[split]
    args=['backtesting','--config',str(folder/'config.json'),'--strategy','V4TargetWeights','--strategy-path',str(ROOT/'freqtrade_userdir/strategies'),
          '--datadir',str(BT/'data'),'--timerange',timerange,'--fee',str(CFG['cost_per_side'][cost]),
          '--export','trades','--export-filename',str(folder/'result.json'),'--cache','none']
    ledger=Ledger(OUT/'ledger.jsonl');ledger.append(dict(kind='BACKTEST_STARTED',model=model,cost=cost,split=split,git_sha=git_sha(ROOT)))
    result=run_offline(ROOT,args,folder/'run.log');write_json(receipt,result)
    ledger.append(dict(kind='BACKTEST_COMPLETED' if result['exit_code']==0 else 'BACKTEST_FAILED',model=model,cost=cost,split=split,**result))
    print(model,cost,split,result['exit_code'],flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--offline',action='store_true');parser.add_argument('--no-external-calls',action='store_true');args=parser.parse_args()
    export_data()
    predictions=pd.read_parquet(OUT/'dev_predictions.parquet') if (OUT/'dev_predictions.parquet').exists() else None
    models=['rule'] if args.smoke else ['buy_hold','sma200','rule','direction_only','ridge','elasticnet','lightgbm']+[f'sidgtn_seed{s}' for s in CFG['seeds']]
    results=[]
    for model in models:
        for cost in (['BASE'] if args.smoke else CFG['cost_per_side']):
            r=backtest(model,cost,predictions=predictions);results.append(r)
            if r['exit_code']!=0:raise SystemExit(r['exit_code'])
    write_json(OUT/'freqtrade/runs.json',results)
if __name__=='__main__':main()
