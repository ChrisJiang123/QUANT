"""Reports use the Freqtrade exported wallets and fills, not a second simulator."""
import io,json,re,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from quant_platform.research.governance import write_json,digest_file
from quant_platform.research.statistics import sharpe,paired_bootstrap,deflated_sharpe
from v4_bootstrap import ROOT
OUT=ROOT/'experiments/v4'

def archive_for(folder):
    target=folder/'engine_result.zip'
    if target.exists():return target
    log=(folder/'run.log').read_text(encoding='utf-8',errors='replace')
    matches=re.findall(r'backtest-result-[\d_-]+\.meta\.json',log)
    if not matches:raise ValueError('engine export absent: '+str(folder))
    source=ROOT/'freqtrade_userdir/backtest_results'/matches[-1].replace('.meta.json','.zip')
    shutil.copy2(source,target)
    return target

def load_engine(folder):
    path=archive_for(folder)
    with zipfile.ZipFile(path) as z:
        name=next(n for n in z.namelist() if n.endswith('.json') and not n.endswith('_config.json') and '_V4TargetWeights' not in n)
        s=json.loads(z.read(name))['strategy']['V4TargetWeights']
        wallet=pd.read_feather(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('_wallet.feather')))))
    return s,wallet,path

def summarize(folder):
    s,w,path=load_engine(folder)
    equity=w.groupby('date').total_quote.sum().sort_index()
    r=equity.pct_change().fillna(0)
    noncash=w[w.currency!='USD'].groupby('date').total_quote.sum().reindex(equity.index,fill_value=0)
    exposure=noncash/equity
    equity_frame=pd.DataFrame({'date':equity.index,'equity':equity.to_numpy(),'return':r.to_numpy(),'exposure':exposure.to_numpy()})
    equity_frame.to_parquet(folder/'engine_equity.parquet',index=False)
    orders=[]
    for trade in s['trades']:
        for o in trade['orders']:
            orders.append(dict(pair=trade['pair'],**o))
    order_frame=pd.DataFrame(orders)
    if len(order_frame):
        order_frame=order_frame.drop_duplicates(subset=['pair','order_filled_timestamp','ft_order_side','amount','safe_price'])
        order_frame.to_parquet(folder/'engine_fills.parquet',index=False)
        turnover=float((order_frame.amount*order_frame.safe_price).sum()/10000)
        fee=float((order_frame.amount*order_frame.safe_price*float(json.loads((folder/'config.json').read_text()).get('fee',0))).sum())
    else:turnover=0.;fee=0.
    # Cost rate comes from the actual recorded CLI, not exchange defaults.
    receipt=json.loads((folder/'receipt.json').read_text());cmd=receipt['command'];rate=float(cmd[cmd.index('--fee')+1])
    fee=float((order_frame.amount*order_frame.safe_price).sum()*rate) if len(order_frame) else 0.
    yearly=[]
    for year,g in equity_frame.groupby(equity_frame.date.dt.year):
        y=g['return'].to_numpy();curve=np.cumprod(1+y)
        yearly.append(dict(year=int(year),net_return=float(curve[-1]-1),sharpe=sharpe(y),max_drawdown=float((curve/np.maximum.accumulate(curve)-1).min()),days=len(y)))
    metrics=dict(schema_version=1,engine='Freqtrade 2026.10',engine_archive=str(path.relative_to(ROOT)),archive_sha256=digest_file(path),
                 net_return=float(s['profit_total']),final_balance=float(s['final_balance']),
                 mark_to_market_net_return=float(equity.iloc[-1]/10000-1),sharpe=sharpe(r.to_numpy()),
                 max_drawdown=float((equity/equity.cummax()-1).min()),trades=int(s['total_trades']),filled_orders=len(order_frame),
                 rejected_signals=int(s['rejected_signals']),turnover_notional_over_initial=turnover,effective_cost_paid=fee,
                 effective_cost_per_side=rate,mean_exposure=float(exposure.mean()),days=len(equity),independent_28d_windows=len(equity)//28,
                 annual=yearly,engine_wallet_metrics=s.get('wallet_stats',{}),limitations=['daily OHLC next-open fills','contemporary pair precision snapshot applied historically','no orderbook history','modeled effective spread/slippage','native network guard, no Docker network namespace'])
    write_json(folder/'metrics.json',metrics)
    return metrics,equity_frame,order_frame

def reconciliation(folder,tolerance=.06):
    s,w,path=load_engine(folder);cfg=json.loads((folder/'config.json').read_text());signals=Path(cfg['v4_signals'])
    total=w.groupby('date').total_quote.sum();rows=[]
    # Engine wallet exports are BEFORE the day's fills. Reconcile post-fill
    # holdings from engine order records at that same day's OPEN, without PnL reporting.
    _,_,orders=summarize(folder)
    orders['date']=pd.to_datetime(orders.order_filled_timestamp,unit='ms',utc=True)
    orders=orders.sort_values('date')
    prices={p:pd.read_feather(Path(cfg['v4_data_dir'])/(p.replace('/','_')+'-1d.feather')).set_index('date').open for p in ['BTC/USD','ETH/USD']}
    receipt=json.loads((folder/'receipt.json').read_text());cmd=receipt['command'];fee=float(cmd[cmd.index('--fee')+1])
    cash=10000.;quantities={'BTC/USD':0.,'ETH/USD':0.};actuals={}
    for date in total.index:
        day=orders[orders.date==date]
        for _,o in day.iterrows():
            notional=float(o.amount*o.safe_price)
            if o.ft_order_side=='buy':quantities[o.pair]+=o.amount;cash-=notional*(1+fee)
            else:quantities[o.pair]-=o.amount;cash+=notional*(1-fee)
        marks={p:quantities[p]*float(prices[p].get(date,np.nan)) for p in quantities}
        equity=cash+sum(marks.values())
        actuals[date]={p:marks[p]/equity for p in quantities}
    # Compare only decision fill times. Forced terminal exits reported separately.
    for pair in ['BTC/USD','ETH/USD']:
        sym=pair.split('/')[0];sig=pd.read_parquet(signals/(pair.replace('/','_')+'.parquet'))
        sig=sig[sig.decision&~sig.gap_mask].set_index('earliest_executable_at')
        asset=w[w.currency==sym].set_index('date').total_quote
        for date,expected in sig.target_weight.items():
            if date not in total.index:continue
            actual=float(actuals[date][pair])
            rows.append(dict(pair=pair,date=str(date),target=float(expected),actual=actual,error=abs(actual-float(expected))))
    result=pd.DataFrame(rows)
    result.to_parquet(folder/'reconciliation.parquet',index=False)
    metric=dict(schema_version=1,tolerance=tolerance,max_error=float(result.error.max()) if len(result) else None,
                compared_weights=len(result),violations=int((result.error>tolerance).sum()) if len(result) else 0,
                passed=bool(len(result) and (result.error<=tolerance).all()),basis='post-fill engine orders at actual next-open prices; accounting checker only, no performance output')
    write_json(folder/'reconciliation_metrics.json',metric)
    return metric

def collect():
    metrics={};curves={}
    for receipt in (OUT/'freqtrade_v2').glob('*/*/*/receipt.json'):
        if json.loads(receipt.read_text())['exit_code']!=0:continue
        folder=receipt.parent;key='/'.join(folder.relative_to(OUT/'freqtrade_v2').parts)
        m,c,o=summarize(folder);metrics[key]=m;curves[key]=c
    write_json(OUT/'backtest_summary/metrics.json',metrics)
    return metrics,curves
if __name__=='__main__':collect()
