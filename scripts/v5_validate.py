"""Generate every conclusion from frozen predictions and complete DEV replays."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,CFG,read,registered,status
from quant_platform.research.statistics import dm_hac,holm,paired_bootstrap,deflated_sharpe,sharpe
from quant_platform.research.governance import write_json,digest_file,now

def returns(name):
    h=pd.read_parquet(OUT/'backtests_v2'/name/'equity.parquet');eq=h.equity.to_numpy()
    return np.diff(np.r_[7000.,eq])/np.r_[7000.,eq[:-1]]

def forecast(f):
    results={};primary=[];pvalues=[]
    for variant,g in f.groupby('variant'):
        # A seed ensemble is a registered descriptive aggregation, never the
        # best seed. Individual seed metrics are also retained below.
        numeric=[c for c in g if any(c.startswith(p) for p in ['target_','q10_','q50_','q90_','tail_','direction_probability_','volatility_'])]
        ensemble=g.groupby(['pair','execution_at','fold'],as_index=False)[numeric].mean()
        for h in CFG['horizons']:
            end=pd.to_datetime(ensemble.execution_at,utc=True)+pd.Timedelta(days=h)
            cutoff=pd.to_datetime(ensemble.fold.str.replace('wf',''),format='%Y',utc=True)+pd.DateOffset(years=1)
            part=ensemble[(end<cutoff)&ensemble[f'target_{h}'].notna()].copy()
            actual=part[f'target_{h}'];pred=part[f'q50_{h}'];err=(actual-pred).abs();zero=actual.abs()
            diff=part.assign(loss_improvement=zero-err).groupby('execution_at').loss_improvement.mean().to_numpy()
            dm=dm_hac(diff,h);prob=part[f'direction_probability_{h}'];truth=(actual>0).astype(float)
            d=dict(samples=len(part),n_dates=len(diff),independent_windows=len(diff)//h,mae=float(err.mean()),zero_mae=float(zero.mean()),mae_skill=float(1-err.mean()/zero.mean()),dm=dm,
                brier=float(((prob-truth)**2).mean()),coverage80=float(((actual>=part[f'q10_{h}'])&(actual<=part[f'q90_{h}'])).mean()),
                tail_exceedance05=float((actual<part[f'tail_{h}']).mean()),
                predicted_vol_mean=float(part[f'volatility_{h}'].mean()),actual_vol_mean=float(part[f'target_vol_{h}'].mean()),
                seed_mae=[dict(seed=int(seed),mae=float((z[f'target_{h}']-z[f'q50_{h}']).abs().mean())) for seed,z in g[pd.to_datetime(g.execution_at,utc=True)+pd.Timedelta(days=h)<pd.to_datetime(g.fold.str.replace('wf',''),format='%Y',utc=True)+pd.DateOffset(years=1)].groupby('seed')])
            key=f'{variant}/{h}';results[key]=d;primary.append(key);pvalues.append(dm['p_one_sided'] if dm['p_one_sided'] is not None else 1.)
    for key,p in zip(primary,holm(pvalues)):results[key].update(p_holm=p,pass_horizon=results[key]['mae_skill']>0 and p<.05)
    return results

def main():
    registered();plan=read(OUT/'backtest_plan.json');allmetrics={}
    for s in plan['trials']:
        folder=OUT/'backtests_v2'/s['name'];done=read(folder/'complete.json')
        if done['fingerprint']!=plan['fingerprint'] or digest_file(folder/'metrics.json')!=done['metrics_hash']:raise ValueError('comparison not complete or altered')
        allmetrics[s['name']]=read(folder/'metrics.json')
    for p in (OUT/'backtests_v2').glob('matched_*/metrics.json'):
        done=read(p.parent/'complete.json')
        if digest_file(p)!=done['metrics_hash']:raise ValueError('matched metrics changed')
        allmetrics[p.parent.name]=read(p)
    f=pd.read_parquet(OUT/'model_predictions.parquet');f=f[f.fold.str.startswith('wf')]
    forecasts=forecast(f);families={};increments={}
    trials=len(allmetrics)+60+len(read(OUT/'legacy_diagnostic.json')['learning_curves'])+len(read(OUT/'mapping_diagnostics.json')['raw_head_replays'])+24
    for variant in CFG['model']['variants']:
        seeds=[]
        for seed in CFG['seeds']:
            name=f'{variant}_{seed}_mm_BASE';m=allmetrics[name];a=returns(name)
            baseline_names=[f'rule_{mode}_BASE' for mode in ['buy_hold','long_only','asymmetric','filtered','carry']]
            baseline_names += [n for n in allmetrics if n.startswith(f'matched_{variant}_') and n.endswith('_BASE')]
            strongest=max(baseline_names,key=lambda n:allmetrics[n]['sharpe']);b=returns(strongest);ci=paired_bootstrap(a,b,draws=1000)
            delta=m['sharpe']-allmetrics[strongest]['sharpe'];dsr=deflated_sharpe(a,trials)
            stress_baselines=[f'rule_{mode}_STRESS' for mode in ['buy_hold','long_only','asymmetric','filtered','carry']]
            stress=max(stress_baselines,key=lambda n:allmetrics[n]['sharpe'])
            sd=allmetrics[f'{variant}_{seed}_mm_STRESS']['sharpe']-allmetrics[stress]['sharpe']
            history=pd.read_parquet(OUT/'backtests_v2'/name/'equity.parquet')
            regimes={regime:sharpe(a[history.market_regime.to_numpy()==regime])-sharpe(b[history.market_regime.to_numpy()==regime]) for regime in sorted(history.market_regime.unique())}
            regime_pass=sum(delta>=0 for delta in regimes.values())>len(regimes)/2
            g2=delta>=.20 and ci['lower90'] is not None and ci['lower90']>0 and dsr is not None and dsr>=.95
            survival=all(allmetrics[f'{variant}_{seed}_mm_{cost}']['survival_pass'] for cost in CFG['costs'])
            seeds.append(dict(seed=seed,baseline=strongest,sharpe=m['sharpe'],delta=delta,ci=ci,DSR=dsr,G2=g2,stress_delta=sd,regime_deltas=regimes,majority_segments_nonnegative=regime_pass,survival=survival,realized_vol=m['realized_vol']))
        g1=all(forecasts[f'{variant}/{h}']['pass_horizon'] for h in [14,28]);median_delta=float(np.median([s['delta'] for s in seeds]))
        # Conservative requirement: at least 3/5 individually pass G2 plus
        # median/worst/STRESS consistency. No best seed promotion.
        g2=sum(s['G2'] for s in seeds)>=3
        g3=min(s['delta'] for s in seeds)>=0 and np.median([s['stress_delta'] for s in seeds])>0 and sum(s['majority_segments_nonnegative'] for s in seeds)>=3
        survival=all(s['survival'] for s in seeds)
        families[variant]=dict(G1=g1,G2=g2,G3=g3,G4=False,G4_status='INCONCLUSIVE_DATA_NO_UNTOUCHED_FINAL',survival_scenario=survival,us_survival_verified=False,
            median_seed_delta=median_delta,worst_seed_delta=min(s['delta'] for s in seeds),mean_seed_delta=float(np.mean([s['delta'] for s in seeds])),seeds=seeds,
            verdict='NO_GO_ALPHA' if not(g1 and g2 and g3 and survival) else 'INCONCLUSIVE_DATA')
        if variant!='core':
            branch=[]
            for seed in CFG['seeds']:
                a=returns(f'{variant}_{seed}_mm_BASE');b=returns(f'core_{seed}_mm_BASE');ci=paired_bootstrap(a,b,draws=1000)
                branch.append(dict(seed=seed,delta=sharpe(a)-sharpe(b),ci=ci,stress_delta=allmetrics[f'{variant}_{seed}_mm_STRESS']['sharpe']-allmetrics[f'core_{seed}_mm_STRESS']['sharpe']))
            forecast_better=all(forecasts[f'{variant}/{h}']['mae']<forecasts[f'core/{h}']['mae'] for h in [14,28])
            qualifies=forecast_better and np.median([x['ci']['lower90'] for x in branch])>0 and np.median([x['stress_delta'] for x in branch])>0
            increments[variant]=dict(primary_mae_improve=forecast_better,seed_results=branch,fusion_qualified=bool(qualifies))
    # Dynamic research only becomes executable once the evidence gate actually
    # passes. Never silently enable it in production/replay configuration.
    dynamic=any(x['fusion_qualified'] for x in increments.values())
    result=dict(generated_at=now(),scope='DEV repeated/consumed spot scenarios; no U.S. perp Alpha claim',forecast=forecasts,families=families,branch_increments=increments,
        dynamic_fusion_qualified=dynamic,dynamic_fusion_executed=False,trial_count_for_DSR=trials,all_metrics=allmetrics,
        final=dict(status='REFUSED_NO_CLEAN_FINAL',independent_clean_windows=0,required=12,consumed_flag_sha256=digest_file(ROOT/'experiments/FINAL_CONSUMED.json')),
        multiple_testing='Holm all 3 variants x 5 horizons, daily date aggregation; DM lag h-1; paired28d blocks; all five seeds shown',
        verdict='NO_GO_ALPHA' if all(v['verdict']=='NO_GO_ALPHA' for v in families.values()) else 'INCONCLUSIVE_DATA')
    write_json(OUT/'validation.json',result)
    for rid in ['V5-VAL','V5-DIAG']:status(rid,'IMPLEMENTED_AND_TESTED',['experiments/v5/validation.json','experiments/v5/legacy_diagnostic.json'],financial=result['verdict'])
    status('V5-FUSION','IMPLEMENTED_AND_TESTED',['experiments/v5/validation.json','experiments/v5/tests-pre-finance.xml'],run='GATED_DISABLED' if not dynamic else 'QUALIFIED_NOT_RUN',financial='NO_INCREMENTAL_EVIDENCE' if not dynamic else 'PENDING_DYNAMIC_RESEARCH')
    print(json.dumps(dict(verdict=result['verdict'],fusion_qualified=dynamic,families={k:{t:v[t] for t in ['G1','G2','G3','G4','median_seed_delta','verdict']} for k,v in families.items()})))

if __name__=='__main__':main()
