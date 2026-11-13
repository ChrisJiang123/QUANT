import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from quant_platform.research.statistics import sharpe,paired_bootstrap,deflated_sharpe
from quant_platform.research.governance import write_json,Ledger
from v4_engine_metrics import collect,OUT
from v4_bootstrap import ROOT

def main():
    metrics,curves=collect();forecast=json.loads((OUT/'validation/metrics.json').read_text())['forecasts']
    baselines=['buy_hold','sma200','rule','ridge','elasticnet','lightgbm']
    eligible=[n for n in baselines if f'DEV/{n}/BASE' in metrics]
    best=max(eligible,key=lambda n:metrics[f'DEV/{n}/BASE']['sharpe']) if eligible else None
    ledger=Ledger(OUT/'ledger.jsonl');ledger.verify()
    trials=sum(r['record'].get('kind') in {'TRIAL_STARTED','BACKTEST_STARTED'} for r in ledger.rows())
    gates={}
    for seed in [11,22,33,44,55]:
        name=f'sidgtn_seed{seed}';g1=all(forecast[f'{name}/{h}']['G1_horizon_pass'] for h in [14,28])
        result=dict(G1=g1,G2=False,G3=False,G4=False)
        key=f'DEV/{name}/BASE'
        if best and key in curves:
            a=curves[key].set_index('date')['return'];b=curves[f'DEV/{best}/BASE'].set_index('date')['return'];align=pd.concat([a,b],axis=1).dropna()
            ci=paired_bootstrap(align.iloc[:,0],align.iloc[:,1]);dsr=deflated_sharpe(a,trials)
            delta=metrics[key]['sharpe']-metrics[f'DEV/{best}/BASE']['sharpe']
            sk=f'DEV/{name}/STRESS';bk=f'DEV/{best}/STRESS'
            stress_delta=metrics[sk]['sharpe']-metrics[bk]['sharpe'] if sk in metrics and bk in metrics else None
            result.update(sharpe_delta=delta,bootstrap=ci,deflated_sharpe=dsr,trial_count=trials,stress_delta=stress_delta,
                          G2=bool(delta>=.2 and ci['lower90'] is not None and ci['lower90']>0 and dsr is not None and dsr>=.95))
        gates[name]=result
    splits=json.loads((ROOT/'experiments/SPLITS_V4.json').read_text());fresh=splits['FRESH_HOLDOUT']['independent_28d_windows']
    seed_deltas=[g.get('sharpe_delta',-float('inf')) for g in gates.values()]
    median_g2=sum(g['G2'] for g in gates.values())>=3
    majority_states={}
    bars=pd.read_parquet(OUT/'bars.parquet');btc=bars[bars.pair=='XBTUSD'].sort_values('candle_ts').copy()
    btc['trend_up']=btc.close>btc.close.rolling(200).mean();btc['rv']=btc.close.pct_change(fill_method=None).rolling(28).std()
    q1=btc.rv.expanding(min_periods=200).quantile(1/3);q2=btc.rv.expanding(min_periods=200).quantile(2/3)
    btc['vol_state']=np.where(btc.rv<q1,'LOW',np.where(btc.rv<q2,'MID','HIGH'))
    btc['state']=np.where(btc.trend_up,'UP_','DOWN_')+btc.vol_state
    states=btc[['available_at','state']].sort_values('available_at')
    for name,g in gates.items():
        key=f'DEV/{name}/BASE'
        if best and key in curves:
            a=curves[key][['date','return']].merge(curves[f'DEV/{best}/BASE'][['date','return']],on='date',suffixes=('_model','_baseline'))
            a=pd.merge_asof(a.sort_values('date'),states,left_on='date',right_on='available_at',direction='backward')
            by={str(s):dict(samples=len(z),model_sharpe=sharpe(z.return_model),baseline_sharpe=sharpe(z.return_baseline),
                           delta=sharpe(z.return_model)-sharpe(z.return_baseline)) for s,z in a.groupby('state')}
            majority=sum(v['delta']>=0 for v in by.values())>len(by)/2
            g['states']=by;g['G3']=bool(g.get('stress_delta') is not None and g['stress_delta']>0 and median_g2 and min(seed_deltas)>=0 and majority)
            g['G4']=False if fresh<12 else None
    # No promotion when any preregistered gate fails; deviation is an additional veto.
    candidate=any(g['G1'] and g['G2'] and g['G3'] for g in gates.values())
    alpha='INCONCLUSIVE_DATA' if candidate and fresh<12 else 'ALPHA_CANDIDATE' if candidate else 'NO_GO_ALPHA'
    if candidate:alpha='NO_GO_ALPHA' # disclosed auxiliary-loss preregistration deviation vetoes promotion
    write_json(OUT/'scorecard/metrics.json',dict(schema_version=1,alpha=alpha,best_simple_baseline=best,gates=gates,
                fresh_independent_windows=fresh,minimum_independent_windows=12,
                required_additional_days=max(0,12*28-splits['FRESH_HOLDOUT']['calendar_days']),
                trials=trials,protocol_deviation_veto=True,baseline_and_strategy_metrics=metrics))
    # Ten-year research index: chain engine returns from separate evidence segments.
    years=[];recent=[];pressure=[]
    for cost in ['OPTIMISTIC','BASE','STRESS']:
        for model in ['buy_hold','equal_exposure_hold','sma200','rule','ridge','lightgbm','sidgtn_seed11']:
            parts=[]
            for split in ['DEV','TOUCHED_AUDIT','FRESH_HOLDOUT']:
                key=f'{split}/{model}/{cost}'
                if key in curves:
                    z=curves[key].copy();z['segment']=split;parts.append(z)
            if not parts:continue
            allc=pd.concat(parts).sort_values('date');allc['research_index']=(1+allc['return']).cumprod()
            allc.to_parquet(OUT/f'tenyear_{model}_{cost}.parquet',index=False)
            for (year,segment),z in allc.groupby([allc.date.dt.year,'segment']):
                r=z['return'].to_numpy();row=dict(model=model,cost=cost,year=int(year),segment=segment,days=len(z),
                    net_return=float(np.prod(1+r)-1),sharpe=sharpe(r),mean_exposure=float(z.exposure.mean()),
                    insufficient_early_training=bool(year<=2018 and model in {'ridge','lightgbm','sidgtn_seed11'}))
                years.append(row)
                if year>=2024:recent.append(row)
            if model=='buy_hold' and cost=='BASE':
                curve=allc.set_index('date').research_index;draw=curve/curve.cummax()-1;used=[]
                for trough in draw.sort_values().index:
                    peak=curve.loc[:trough].idxmax()
                    if any(not (trough<a or peak>b) for a,b in used):continue
                    used.append((peak,trough));pressure.append(dict(start=str(peak),trough=str(trough),drawdown=float(draw.loc[trough])))
                    if len(pressure)==5:break
    write_json(OUT/'tenyear/metrics.json',dict(schema_version=1,annual=years,recent_three_years=recent,deepest_buy_hold_drawdowns=pressure,
                   limitations=['early learned-model samples insufficient','strategy designed after observing history','market structure changes',
                                'separate account resets stitched as a research index, not a continuously investable equity curve',
                                'purge/embargo calendar gaps shown explicitly; no fabricated returns','L1 funding blocked, event publication history unverified'],
                   evidence_segments=splits))
    print(alpha)
if __name__=='__main__':main()
