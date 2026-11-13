"""Additional audit comparisons; all returns originate from Freqtrade exports."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from v4_train import dataset
from quant_platform.research.governance import write_json,Ledger
from quant_platform.research.statistics import sharpe,paired_bootstrap,deflated_sharpe,forecast_metrics
from v4_engine_metrics import collect,OUT

def main():
    metrics,curves=collect();results={};state_rows=[];matched=[];pressure=[];risk=[]
    best=max(['ridge','elasticnet','lightgbm'],key=lambda n:metrics[f'DEV/{n}/BASE']['sharpe'])
    trials=sum(row['record'].get('kind') in {'TRIAL_STARTED','BACKTEST_STARTED'} for row in Ledger(OUT/'ledger.jsonl').rows())
    for seed in [11,22,33,44,55]:
        name=f'sidgtn_seed{seed}';a=curves[f'DEV/{name}/BASE'].set_index('date')['return'];b=curves[f'DEV/{best}/BASE'].set_index('date')['return'];z=pd.concat([a,b],axis=1).dropna()
        delta=sharpe(z.iloc[:,0])-sharpe(z.iloc[:,1]);ci=paired_bootstrap(z.iloc[:,0],z.iloc[:,1]);dsr=deflated_sharpe(z.iloc[:,0],trials)
        stress=metrics[f'DEV/{name}/STRESS']['sharpe']-metrics[f'DEV/{best}/STRESS']['sharpe']
        results[name]=dict(best_simple_ml=best,base_sharpe_delta=delta,bootstrap=ci,dsr=dsr,stress_delta=stress,
                         G2=bool(delta>=.2 and ci['lower90'] is not None and ci['lower90']>0 and dsr is not None and dsr>=.95))
    consistent=sum(v['G2'] for v in results.values())>=3 and min(v['base_sharpe_delta'] for v in results.values())>=0
    for v in results.values():v['G3_seed_and_cost_component']=bool(consistent and v['stress_delta']>0)
    f=pd.read_parquet(OUT/'features.parquet');btc=f[f.pair=='XBTUSD'].sort_values('decision_at').copy()
    btc['trend']=np.where(btc.close>btc.close.rolling(200).mean(),'UP','DOWN');q1=btc.vol28.expanding(200).quantile(1/3);q2=btc.vol28.expanding(200).quantile(2/3)
    btc['regime']=btc.trend+'_'+np.where(btc.vol28<q1,'LOW',np.where(btc.vol28<q2,'MID','HIGH'))
    regimes=btc[['decision_at','regime']].sort_values('decision_at')
    for key,curve in curves.items():
        split,model,cost=key.split('/')
        if model not in {'buy_hold','equal_exposure_hold','sma200','rule','ridge','lightgbm','sidgtn_seed11','equal_hold_ridge','equal_hold_lightgbm','equal_hold_sidgtn_seed11'}:continue
        annotated=pd.merge_asof(curve.sort_values('date'),regimes,left_on='date',right_on='decision_at',direction='backward')
        for state,g in annotated.groupby('regime'):
            state_rows.append(dict(split=split,model=model,cost=cost,regime=state,days=len(g),sharpe=sharpe(g['return']),net_return=float(np.prod(1+g['return'])-1)))
        r=curve['return'].to_numpy();sd=float(np.std(r,ddof=1));negative=r[r<0]
        risk.append(dict(key=key,annualized_volatility=sd*np.sqrt(365),worst_engine_wallet_day=float(r.min()),
                         sortino=float(np.sqrt(365)*r.mean()/np.sqrt(np.mean(np.minimum(r,0)**2))) if len(negative) else None,
                         cash_fraction_mean=float(1-curve.exposure.mean()),return_observations=len(r)))
    for split in ['DEV','TOUCHED_AUDIT','FRESH_HOLDOUT']:
        for model,baseline in [('rule','equal_exposure_hold'),('ridge','equal_hold_ridge'),('lightgbm','equal_hold_lightgbm'),('sidgtn_seed11','equal_hold_sidgtn_seed11')]:
            for cost in ['OPTIMISTIC','BASE','STRESS']:
                a=f'{split}/{model}/{cost}';b=f'{split}/{baseline}/{cost}'
                if a not in metrics or b not in metrics:continue
                matched.append(dict(split=split,model=model,benchmark=baseline,cost=cost,model_sharpe=metrics[a]['sharpe'],
                      benchmark_sharpe=metrics[b]['sharpe'],sharpe_delta=metrics[a]['sharpe']-metrics[b]['sharpe'],
                      strategy_exposure=metrics[a]['mean_exposure'],benchmark_realized_exposure=metrics[b]['mean_exposure']))
    ten=json.loads((OUT/'tenyear/metrics.json').read_text())
    for window in ten['deepest_buy_hold_drawdowns']:
        start=pd.Timestamp(window['start']);end=pd.Timestamp(window['trough'])
        for model in ['buy_hold','rule','ridge','lightgbm','sidgtn_seed11']:
            parts=[v for k,v in curves.items() if k.split('/')[1:]==[model,'BASE']]
            if not parts:continue
            c=pd.concat(parts).sort_values('date');g=c[(c.date>=start)&(c.date<=end)]
            pressure.append(dict(start=str(start),end=str(end),model=model,days=len(g),net_return=float(np.prod(1+g['return'])-1)))
    p=pd.read_parquet(OUT/'dev_predictions.parquet');p=p[p.metric_eligible];asset_metrics={pair:forecast_metrics(g) for pair,g in p.groupby('pair')}
    m,_,vol=dataset();volmetrics=[]
    for (name,h),g in p[p.model.str.startswith('sidgtn')].groupby(['model','horizon']):
        actual=vol[g.sample_id.to_numpy(),0 if h==14 else 1];naive=m.iloc[g.sample_id.to_numpy()].vol28.to_numpy()
        mae=float(np.mean(np.abs(g.forecast_vol.to_numpy()-actual)));zero=float(np.mean(np.abs(naive-actual)))
        volmetrics.append(dict(model=name,horizon=int(h),volatility_mae=mae,persistence_mae=zero,skill_vs_persistence=1-mae/zero))
    comparisons={}
    for label,a,b in [('H1_trend','rule','buy_hold'),('H2_vol_target','rule','direction_only')]:
        left=curves[f'DEV/{a}/BASE'].set_index('date')['return'];right=curves[f'DEV/{b}/BASE'].set_index('date')['return'];z=pd.concat([left,right],axis=1).dropna()
        delta=sharpe(z.iloc[:,0])-sharpe(z.iloc[:,1]);ci=paired_bootstrap(z.iloc[:,0],z.iloc[:,1]);comparisons[label]=dict(sharpe_delta=delta,bootstrap=ci,
             interpretation='SUPPORTED_IN_DEV_ONLY' if delta>0 and ci['lower90'] is not None and ci['lower90']>0 else 'NOT_CONFIRMED_IN_THIS_RUN')
    h3=json.loads((OUT/'h3/metrics.json').read_text())
    comparisons['H3_crowding']=dict(result='NOT_CONFIRMED_IN_THIS_RUN' if not h3['comparisons']['BASE']['confirmed'] else 'SUPPORTED_IN_DEV_ONLY',metrics=h3['comparisons'])
    comparisons['H4_calendar']={'result':'INCONCLUSIVE_DATA','reason':'historical announced_at not proven; group masked'}
    comparisons['H5_attention']={'result':'EXPLORATORY_ONLY','reason':'lagged descriptive association, no primary gate or causal claim'}
    write_json(OUT/'extended/metrics.json',dict(schema_version=1,sidgtn_vs_simple_ml=results,asset_forecasts=asset_metrics,
           regimes=state_rows,matched_exposure=matched,pressure_windows=pressure,risk=risk,volatility_prediction=volmetrics,hypotheses=comparisons,
           limitations=['same historical design caveats as primary scorecard','short fresh-anchor period remains inconclusive even when follow-up wallet dates span a longer period']))
    print('EXTENDED_AUDIT_METRICS_FROM_ENGINE_EXPORTS')
if __name__=='__main__':main()
