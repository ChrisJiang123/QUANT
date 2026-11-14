import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.features.registry import PRIMARY
from quant_platform.features.store import FeatureStore
from quant_platform.data.prospective import ProspectiveSource
from quant_platform.research.governance import write_json,digest_file,now,Ledger
from quant_platform.research.statistics import forecast_metrics,sharpe,paired_bootstrap
from v4_backtest import backtest,CFG
from v4_engine_metrics import summarize
OUT=ROOT/'experiments/v4'

def normalize_sources():
    public=OUT/'public';sources={};coverage=[]
    dvol=[]
    for path in [public/'deribit_dvol.raw',*sorted(public.glob('deribit_dvol_page*.raw'))]:
        if not path.exists():continue
        data=json.loads(path.read_text()).get('result',{}).get('data',[]);dvol.extend(data)
    if dvol:
        f=pd.DataFrame(dvol,columns=['ts','open','high','low','value']).drop_duplicates('ts').sort_values('ts')
        f['observed_at']=pd.to_datetime(f.ts,unit='ms',utc=True)+pd.Timedelta(days=1);f['value']=f.value/100
        sources['dvol']=ProspectiveSource('dvol').normalize(f[['observed_at','value']])
    path=public/'fear_greed.raw'
    if path.exists():
        f=pd.DataFrame(json.loads(path.read_text())['data']);f['observed_at']=pd.to_datetime(f.timestamp.astype('int64'),unit='s',utc=True);f['value']=pd.to_numeric(f.value)
        sources['fear_greed']=ProspectiveSource('fear_greed').normalize(f[['observed_at','value']].sort_values('observed_at'))
    path=public/'wikipedia.raw'
    if path.exists():
        f=pd.DataFrame(json.loads(path.read_text())['items']);f['observed_at']=pd.to_datetime(f.timestamp,format='%Y%m%d%H',utc=True)+pd.Timedelta(days=1);f['value']=f.views.astype(float)
        sources['wikipedia']=ProspectiveSource('wikipedia').normalize(f[['observed_at','value']].sort_values('observed_at'))
    for name,f in sources.items():
        folder=OUT/'prospective'/name;folder.mkdir(parents=True,exist_ok=True);f.to_parquet(folder/'values.parquet',index=False)
        coverage.append(dict(source=name,rows=len(f),observed_start=str(f.observed_at.min()),observed_end=str(f.observed_at.max()),
                             availability='UTC source observation plus documented conservative lag; inferred historical publication, not first-seen proof',
                             path=str((folder/'values.parquet').relative_to(ROOT)),sha256=digest_file(folder/'values.parquet')))
    funding=public/'kraken_funding_correct.raw'
    funding_status='BLOCKED'
    if funding.exists():
        raw=json.loads(funding.read_text())
        if raw.get('result')=='success' and raw.get('rates'):
            rates=pd.DataFrame(raw['rates']);rates['observed_at']=pd.to_datetime(rates.timestamp,utc=True).dt.floor('D')+pd.Timedelta(days=1)
            rates['value']=pd.to_numeric(rates.relativeFundingRate)
            daily=rates.groupby('observed_at').agg(value=('value','sum'),hours=('value','size')).reset_index()
            daily=daily[(daily.hours==24)&(daily.observed_at<=pd.Timestamp(now()))]
            sources['funding']=ProspectiveSource('funding').normalize(daily[['observed_at','value']])
            folder=OUT/'prospective/funding';folder.mkdir(parents=True,exist_ok=True);sources['funding'].to_parquet(folder/'values.parquet',index=False)
            f=sources['funding'];coverage.append(dict(source='funding',rows=len(f),observed_start=str(f.observed_at.min()),observed_end=str(f.observed_at.max()),
                 availability='sum of completed hourly relative rates, available at UTC day close',path=str((folder/'values.parquet').relative_to(ROOT)),sha256=digest_file(folder/'values.parquet')))
            funding_status='ACQUIRED_RECENT_HISTORY_NO_DEV_COVERAGE'
    write_json(OUT/'prospective/coverage/metrics.json',dict(schema_version=1,sources=coverage,bitmex='BLOCKED_HTTP_403_NO_RETRY_AROUND_RESTRICTION',
                  kraken_funding=funding_status,basis='UNAVAILABLE_HISTORICAL_PIT',events='BLOCKED_HISTORICAL_ANNOUNCEMENT_TIMES',
                  wikipedia_lag='observation UTC day close plus three days; conservative extra day relative to day-open timestamp'))
    return sources

def main():
    sources=normalize_sources();out=OUT/'l2';out.mkdir(exist_ok=True)
    reg=out/'registration.json'
    if not reg.exists():write_json(reg,dict(schema_version=1,registered_at=now(),scope='DEV_ONLY_L2_INCREMENTAL',
       family='ridge',alpha=CFG['ridge_alpha'],baseline=PRIMARY,additional=['dvol_rolling_rank','iv_minus_rv_rolling_rank'],
       hypotheses=['H2'],primary_gates=False,source='BTC DVOL used as context for both fixed spot assets',
       constraints='same dates, same training cutoff, train-only scaling, no new final test'),immutable=True)
    features=pd.read_parquet(OUT/'features.parquet');frames=[]
    for pair,g in features.groupby('pair'):
        g=g.sort_values('decision_at').copy()
        iv=ProspectiveSource('dvol').join(pd.DatetimeIndex(g.decision_at),sources['dvol'].drop(columns=['available_at','processing_version','source']))
        g['iv']=iv.value.to_numpy();g['dvol_rank']=g.iv.rolling(365,min_periods=60).rank(pct=True)
        g['iv_rv_rank']=(g.iv-g.vol28).rolling(365,min_periods=60).rank(pct=True)
        frames.append(g)
    data=pd.concat(frames,ignore_index=True);predictions=[];ledger=Ledger(OUT/'ledger.jsonl')
    for normfile in sorted((OUT/'training').glob('dev_fold*/normalizer.json')):
        norm=json.loads(normfile.read_text());start=pd.Timestamp(norm['validation_start']);cutoff=pd.Timestamp(norm['train_cutoff'])
        i=int(normfile.parent.name.replace('dev_fold',''));end=pd.Timestamp(f'{2018+i}-01-01',tz='UTC')
        complete=data[PRIMARY+['dvol_rank','iv_rv_rank','y14','y28']].notna().all(axis=1)
        train=complete&(data.label_available28<cutoff);val=complete&(data.decision_at>=start)&(data.decision_at<end)
        if train.sum()<200 or val.sum()==0:continue
        for extra in [False,True]:
            name='ridge_l2_with_dvol' if extra else 'ridge_l2_without_dvol';cols=PRIMARY+(['dvol_rank','iv_rv_rank'] if extra else [])
            ledger.append(dict(kind='TRIAL_STARTED',family='ridge',variant=name,fold=i,scope='L2_DEV_ONLY',registration_sha256=digest_file(reg)))
            scaler=StandardScaler().fit(data.loc[train,cols]);x=scaler.transform(data.loc[train,cols]);xv=scaler.transform(data.loc[val,cols])
            for h in [14,28]:
                model=Ridge(alpha=CFG['ridge_alpha']).fit(x,data.loc[train,f'y{h}']);p=model.predict(xv);subset=data.loc[val]
                predictions.append(pd.DataFrame(dict(model=name,horizon=h,prediction=p,actual=subset[f'y{h}'].to_numpy(),pair=subset.pair.to_numpy(),
                   decision_at=subset.decision_at.to_numpy(),candle_ts=subset.candle_ts.to_numpy(),metric_eligible=(subset.decision_at+pd.Timedelta(days=h)<end).to_numpy())))
            ledger.append(dict(kind='TRIAL_COMPLETED',family='ridge',variant=name,fold=i))
    if not predictions:
        write_json(out/'metrics.json',dict(schema_version=1,status='INCONCLUSIVE_DATA',reason='insufficient same-period L2 training/validation'));return
    allp=pd.concat(predictions,ignore_index=True);allp.to_parquet(out/'predictions.parquet',index=False);costs={};comparisons={}
    for name in sorted(allp.model.unique()):
        for cost in CFG['cost_per_side']:
            result=backtest(name,cost,predictions=allp)
            if result['exit_code']:raise SystemExit(result['exit_code'])
            folder=OUT/'freqtrade_v2/DEV'/name/cost;metric,curve,_=summarize(folder)
            actual_start=allp.decision_at.min();curve=curve[curve.date>=actual_start]
            costs[f'{name}/{cost}']=dict(net_return=metric['net_return'],same_period_sharpe=sharpe(curve['return']),days=len(curve),
                        start=str(actual_start),end=str(curve.date.max()),engine_source=str((folder/'metrics.json').relative_to(ROOT)))
    for cost in CFG['cost_per_side']:
        comparisons[cost]=costs[f'ridge_l2_with_dvol/{cost}']['same_period_sharpe']-costs[f'ridge_l2_without_dvol/{cost}']['same_period_sharpe']
    # Exploratory attention summaries, date-lagged, excluded from primary gates.
    exploration=[]
    for name in ['fear_greed','wikipedia']:
        for pair,g in features.groupby('pair'):
            g=g[g.decision_at<pd.Timestamp('2025-01-01',tz='UTC')].sort_values('decision_at')
            raw=sources[name].drop(columns=['available_at','processing_version','source']);joined=ProspectiveSource(name).join(pd.DatetimeIndex(g.decision_at),raw)
            z=(joined.value-joined.value.rolling(365,min_periods=60).mean())/joined.value.rolling(365,min_periods=60).std()
            actual=g.y28.reset_index(drop=True);valid=z.notna()&actual.notna();rho=float(spearmanr(z[valid],actual[valid]).statistic) if valid.sum()>56 else None
            exploration.append(dict(source=name,pair=pair,overlapping_samples=int(valid.sum()),independent_windows=int(valid.sum())//28,
                                    descriptive_spearman=rho,primary_gate_eligible=False,causal_claim=False))
    write_json(OUT/'sentiment_exploration/metrics.json',dict(schema_version=1,exploratory_only=True,results=exploration))
    write_json(out/'metrics.json',dict(schema_version=1,status='SAME_PERIOD_DEV_ONLY',forecasts=forecast_metrics(allp[allp.metric_eligible]),
                   cost_comparison=costs,incremental_sharpe=comparisons,terminal_reused=False,not_comparable_to_tenyear_headline=True))
    # Actual long-form values, immutable and individually availability stamped.
    f=features.melt(id_vars=['pair','observed_at','available_at','processing_version'],value_vars=PRIMARY,var_name='factor',value_name='value').dropna(subset=['observed_at','available_at'])
    store=FeatureStore(ROOT/'data/v4_feature_store');path=store.append(f)
    write_json(OUT/'feature_store/metrics.json',dict(schema_version=1,values=len(f),path=str(path.relative_to(ROOT)),sha256=digest_file(path)))
    print('PIT_PROSPECTIVE_AND_L2_INCREMENTAL_EVIDENCE_COMPLETE')
if __name__=='__main__':main()
