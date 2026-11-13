"""Fixed, DEV-only crowding filter; historical derivatives data is information only."""
import json
import numpy as np,pandas as pd
from v4_bootstrap import ROOT
from v4_backtest import backtest,CFG,OUT
from v4_engine_metrics import summarize
from quant_platform.research.governance import write_json,now,digest_file
from quant_platform.research.statistics import sharpe,paired_bootstrap

def main():
    out=OUT/'h3';out.mkdir(exist_ok=True);reg=out/'registration.json'
    if not reg.exists():write_json(reg,dict(schema_version=1,registered_at=now(),hypothesis='H3',scope='DEV_ONLY_L2_SAME_PERIOD',
        contract='PI_XBTUSD/PI_ETHUSD fixed inverse-perpetual information source; spot trades only',
        aggregation='median relative rate at UTC day close; avoids summing across changing rate intervals',
        rolling_days=365,min_observations=60,long_crowding_rank=.9,coefficient=.5,control_coefficient=1.,
        primary_gate_eligible=False,terminal_reused=False),immutable=True)
    source=json.loads((OUT/'funding_export/whole_metrics.json').read_text());features=pd.read_parquet(OUT/'features.parquet');parts=[];coverage=[]
    for pair,contract in [('XBTUSD','PI_XBTUSD'),('ETHUSD','PI_ETHUSD')]:
        member=next(m for m in source['members'] if m['name']==f'exports/{contract}.csv')
        f=pd.read_csv(ROOT/member['path']);f['observed_at']=pd.to_datetime(f.timestamp,utc=True).dt.floor('D')+pd.Timedelta(days=1)
        f['value']=pd.to_numeric(f.relative_rate);daily=f.groupby('observed_at').agg(value=('value','median'),count=('value','size')).reset_index()
        daily=daily[(daily['count']>=4)&(daily.observed_at<=pd.Timestamp(now()))]
        daily['available_at']=daily.observed_at;daily['processing_version']='v4-funding-median-rank';daily['pair']=pair
        daily['funding_rank']=daily.value.rolling(365,min_periods=60).rank(pct=True)
        daily.to_parquet(out/(pair+'_pit.parquet'),index=False)
        coverage.append(dict(pair=pair,contract=contract,days=len(daily),start=str(daily.observed_at.min()),end=str(daily.observed_at.max()),
                   source_sha256=member['sha256'],aggregation='daily median relative_rate; not a funding payment or spot cost',observations_per_day_min=int(daily['count'].min()),observations_per_day_max=int(daily['count'].max())))
        g=features[features.pair==pair].sort_values('decision_at').copy()
        joined=pd.merge_asof(g,daily[['available_at','funding_rank']],left_on='decision_at',right_on='available_at',direction='backward',suffixes=('','_funding'))
        joined.loc[joined.decision_at-joined.available_at_funding>pd.Timedelta(days=3),'funding_rank']=np.nan
        joined['score']=sum((joined.close>joined.close.rolling(window).mean()).astype(float) for window in CFG['trend_windows'])/len(CFG['trend_windows'])
        valid=joined.funding_rank.notna()&~joined.missing_mask&joined.y28.notna()&(joined.decision_at<pd.Timestamp('2025-01-01',tz='UTC'))
        for variant in ['control','crowding_filter']:
            s=joined.loc[valid].copy();c=np.where(s.funding_rank>.9,.5,1.) if variant=='crowding_filter' else 1.
            s['prediction']=(s.score*c-.5)*.1;s['actual']=s.y28;s['horizon']=28;s['model']='price_h3_'+variant
            parts.append(s[['pair','candle_ts','decision_at','prediction','actual','horizon','model']])
    predictions=pd.concat(parts,ignore_index=True);predictions.to_parquet(out/'predictions.parquet',index=False);result={};curves={}
    start=predictions.decision_at.min()
    for name in sorted(predictions.model.unique()):
        for cost in CFG['cost_per_side']:
            r=backtest(name,cost,predictions=predictions)
            if r['exit_code']:raise SystemExit(r['exit_code'])
            folder=OUT/'freqtrade_v2/DEV'/name/cost;m,c,_=summarize(folder);c=c[c.date>=start]
            result[f'{name}/{cost}']=dict(net_return=m['net_return'],same_period_sharpe=sharpe(c['return']),days=len(c),start=str(start),end=str(c.date.max()))
            curves[f'{name}/{cost}']=c
    comparison={}
    for cost in CFG['cost_per_side']:
        a=curves[f'price_h3_crowding_filter/{cost}'].set_index('date')['return'];b=curves[f'price_h3_control/{cost}'].set_index('date')['return']
        z=pd.concat([a,b],axis=1).dropna();delta=sharpe(z.iloc[:,0])-sharpe(z.iloc[:,1]);ci=paired_bootstrap(z.iloc[:,0],z.iloc[:,1])
        comparison[cost]=dict(sharpe_delta=delta,bootstrap=ci,confirmed=bool(delta>0 and ci['lower90'] is not None and ci['lower90']>0))
    write_json(out/'metrics.json',dict(schema_version=1,status='SAME_PERIOD_DEV_ONLY',coverage=coverage,comparisons=comparison,strategy_metrics=result,
           source_is_information_only=True,derivatives_traded=False,terminal_reused=False,not_tenyear_L1_evidence=True,
           limitation='publisher historical series and conservative availability inference, not proven historical first-seen vintages'))
    print('H3_CROWDING_FILTER_SAME_PERIOD_COMPLETE')
if __name__=='__main__':main()
