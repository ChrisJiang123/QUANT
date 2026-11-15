"""Registered DEV scenario replay. Spot proxies are never U.S. perp history."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, OUT, CFG, read
from .strategy import Controller, trend_features
from .book import Book
from .engine import Engine
from .instruments import InstrumentSpec

def specifications():
    return {r['pair']:InstrumentSpec(instrument_id=r['instrument_id'],asset=r['pair'],multiplier=r['multiplier'],tick=r['tick'],product_id=r['product_id'],source_available_at=r['source_available_at']) for r in read(OUT/'instrument_specs.json')['instruments']}

def inputs():
    bars=pd.read_parquet(ROOT/'experiments/v4/bars.parquet').sort_values(['pair','candle_ts'])
    daily=pd.read_parquet(OUT/'data/funding_daily.parquet')
    raw=pd.read_parquet(OUT/'data/funding_all.parquet').dropna(subset=['interval_start','rate']).copy()
    rows={};fund={};coverage=[]
    for asset,g in bars.groupby('pair'):
        f=trend_features(g,CFG);d=daily[daily.pair==asset].copy().sort_values('day')
        d=d.set_index('day').reindex(pd.date_range(d.day.min(),d.day.max(),freq='D'))
        rate=d.rate;d['funding_mean7_annual']=rate.rolling(7).mean()*365
        d['funding_rank30']=rate.rolling(30).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1],raw=True)
        d['funding_current']=rate;d['funding_prior_high']=rate.shift().rolling(7).max()
        d['funding_prior_rank']=d.funding_rank30.shift().rolling(7).max()
        d['negative3']=rate.lt(0).rolling(3).sum().eq(3)&rate.notna().rolling(3).sum().eq(3)
        # Observation day midnight is not a usable rate. Join by publication,
        # not by the date printed on an end-of-interval settlement.
        d['available_at']=d.index+pd.Timedelta(days=1,hours=6)
        wanted=['available_at','funding_mean7_annual','funding_rank30','funding_current','funding_prior_high','funding_prior_rank','negative3']
        f=pd.merge_asof(f.sort_values('available_at'),d[wanted].sort_values('available_at'),on='available_at',direction='backward',tolerance=pd.Timedelta(days=2))
        f['funding_negative3_price_up']=f.negative3.fillna(False).astype(bool)&(f.close>f.close.shift(3))
        f['event_coefficient']=1. # No proven historic announcements: disabled.
        rows[asset]={pd.Timestamp(r['execution_at']):r for r in f.to_dict('records')}
        rr=raw[raw.pair==asset].copy();rr['day']=rr.event_at.dt.floor('D')
        # For financial accounting select an actual continuous 24h interval
        # union, not sum(periods) by timestamp day (4h->1h switch is 27h).
        byday={}
        for day,part in rr.groupby('day'):
            candidates=[]
            for linear,p in part.groupby('linear_source'):
                p=p.sort_values('event_at');left=day;right=day+pd.Timedelta(days=1)
                # The midnight interval ending at left belongs to yesterday;
                # require coverage of [left,right) from this day's events plus
                # the event at right. Its charge is processed on next day's
                # timeline against actual exposure, not future signal data.
                allp=rr[(rr.linear_source==linear)&(rr.interval_start<right)&(rr.interval_end>left)].sort_values('interval_start')
                cursor=left;ok=True
                for x in allp.itertuples():
                    start=max(left,x.interval_start);end=min(right,x.interval_end)
                    if start>cursor:ok=False;break
                    cursor=max(cursor,end)
                if ok and cursor>=right:candidates.append((linear,p))
            if candidates:
                linear,p=max(candidates,key=lambda z:z[0]);byday[day]=p
        for day,p in byday.items():
            fund[(asset,day)]=p
        coverage.append(dict(asset=asset,start=str(min(byday)),end=str(max(byday)),complete_days=len(byday),grade='SCENARIO_ONLY',rate_source='linear preferred; inverse earlier; cadence inferred from raw; spot execution'))
    returns=bars.pivot(index='candle_ts',columns='pair',values='close').pct_change(fill_method=None)
    corr=returns.rolling(60,min_periods=40).corr()
    market={day:{r['pair']:r for r in g.to_dict('records')} for day,g in bars.groupby('candle_ts')}
    return rows,market,fund,corr,coverage

def statistics(history,initial=7000.):
    h=pd.DataFrame(history)
    if h.empty:return dict(days=0)
    eq=h.equity.to_numpy();r=np.diff(np.r_[initial,eq])/np.r_[initial,eq[:-1]]
    sd=float(np.std(r,ddof=1));years=len(r)/365.
    return dict(days=len(r),net_return=float(eq[-1]/initial-1),annual_return=float((eq[-1]/initial)**(1/years)-1) if eq[-1]>0 else -1.,
        sharpe=float(np.mean(r)/sd*np.sqrt(365)) if sd>0 else 0.,realized_vol=sd*np.sqrt(365),max_drawdown=float(np.max(1-eq/np.maximum.accumulate(np.r_[initial,eq])[1:])),
        average_gross=float(h.gross.mean()),peak_gross=float(h.gross.max()),average_net=float(h.net.mean()),peak_margin=float(h.virtual_initial_margin.max()),
        liquidations=int(h.liquidations.iloc[-1]),hard_breakers=int(h.hard_breakers.iloc[-1]),stops=int(h.stops.iloc[-1]),independent_28d_windows=len(r)//28)

def direction_attribution(book):
    """Physical net direction, distinct from strategy sleeve attribution."""
    quantity={a:0 for a in book.specs};pnl={'long':0.,'short':0.};funding={'long':0.,'short':0.}
    for event in book.events:
        if event['kind']=='NET_FILL':
            a=event['asset'];old=quantity[a];delta=event['net_contracts'];new=old+delta
            # The independently stored fill has realized PnL and fixed fees.
            # Match using per-asset monotonic iterator below instead of prices.
            quantity[a]=new
        elif event['kind']=='FUNDING_ACCRUED':
            amount=event['amount'];rate=event['effective_rate']
            if rate and amount:funding['long' if -amount/rate>0 else 'short']+=amount
    quantity={a:0 for a in book.specs}
    for f in book.fills:
        a=f['asset'];old=quantity[a];delta=f['contracts'];new=old+delta
        if old:pnl['long' if old>0 else 'short']+=f['realized']
        closing=min(abs(old),abs(delta)) if old*delta<0 else 0
        if closing:pnl['long' if old>0 else 'short']-=f['fixed_fee']*closing/abs(delta)
        opening=abs(delta)-closing
        if opening:pnl['long' if delta>0 else 'short']-=f['fixed_fee']*opening/abs(delta)
        quantity[a]=new
    for a,p in book.physical.items():
        if p.quantity:pnl['long' if p.quantity>0 else 'short']+=p.quantity*book.specs[a].multiplier*(book.mark[a]-p.average)
    for side in pnl:pnl[side]+=funding[side]
    spot=book.spot_realized+sum(q*(book.mark[a]-book.spot_average[a]) for a,q in book.spot.items())-book.costs['spot']
    pnl['long']+=spot
    residual=book.equity()-book.initial-sum(pnl.values())
    return dict(net_pnl=pnl,funding=funding,unallocated_account_fees=residual,scope='physical net sign; spot assigned long; execution friction embedded in fills')

def replay(folder,mode='filtered',cost='BASE',predictions=None,quadrant='rr',mapping='probability',seed=None,variant=None,
           start='2022-01-01',end='2025-01-01',prepared=None,resume=False,stop_after=None,exposure_scale=1.,funding_stress=False,margin_hike=False,favorable_first=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows,market,fund,corr,coverage=prepared or inputs();assets=CFG['traded_assets']
    dates=pd.date_range(start,end,freq='D',inclusive='left',tz='UTC')
    missing=[str(t) for t in dates if any(a not in market.get(t,{}) or (a,t) not in fund or t not in rows[a] for a in assets)]
    if missing:raise ValueError('incomplete common executable/funding interval: '+str(missing[:8]))
    if resume and (folder/'state/engine.json').exists():engine=Engine.load(folder/'state')
    else:engine=Engine(Book(specifications(),CFG['initial_equity'],**CFG['costs'][cost]),CFG)
    control=Controller(CFG,'model' if predictions is not None else mode,cost,tactical_only=mode=='tactical')
    indexed={} if predictions is None else {(pd.Timestamp(r['execution_at']),r['pair']):r for r in predictions.to_dict('records')}
    for count,t in enumerate(dates):
        if engine.last_day and t<=pd.Timestamp(engine.last_day):continue
        if stop_after is not None and count>=stop_after:break
        observed={a:rows[a][t] for a in assets};bars=market[t];prices={a:bars[a]['open'] for a in assets}
        directions={};vols={}
        if predictions is not None:
            for a in assets:
                p=indexed.get((t,a))
                if p is None:raise ValueError('missing model prediction '+str((t,a)))
                if pd.Timestamp(p['available_at'])>=t:raise ValueError('late model output')
                directions[a]=float(p['direction_score'] if mapping=='probability' else p['median_score'])
                vols[a]=max(CFG['vol_floor'],float(p['annual_vol']))
        used_vol=np.array([vols[a] if quadrant[1]=='m' else observed[a]['annual_vol'] for a in assets])
        # Known correlation uses the observation candle, not today's return.
        day=t-pd.Timedelta(days=2)
        c=corr.loc[day].reindex(index=assets,columns=assets).to_numpy() if day in corr.index.get_level_values(0) else np.eye(len(assets))
        c=np.nan_to_num(c,nan=0.);np.fill_diagonal(c,1.)
        covariance=c*used_vol[:,None]*used_vol[None,:]
        plan=control.plan(observed,prices,engine.book,t,directions if quadrant[0]=='m' else None,vols if quadrant[1]=='m' else None,covariance)
        if exposure_scale!=1:
            plan['targets']={k:math.trunc(n*exposure_scale) for k,n in plan['targets'].items()}
            plan['spot_targets']={a:abs(plan['targets'][('carry',a)])*engine.book.specs[a].multiplier for a in assets}
        events=[]
        for a in assets:
            for event in fund[(a,t)].itertuples():
                # All prices here are explicitly spot-open funding-index
                # scenarios. Raw venue units are not guessed to be USD marks.
                events.append(dict(asset=a,event_at=event.event_at,interval_start=event.interval_start,rate=float(event.rate),price=float(prices[a]),event_id=f'{event.source_symbol}:{event.event_at}',stress=cost=='STRESS' or funding_stress))
        changes=[(a,.40,.50) for a in assets] if margin_hike and t==pd.Timestamp('2022-06-13',tz='UTC') else None
        engine.step(t,bars,funding_events=events,margin_changes=changes,favorable_first=favorable_first,**plan)
        if engine.history and engine.history[-1]['at']==str(t):
            score=float(np.mean([r['raw_score'] for r in observed.values()]))
            engine.history[-1]['market_regime']='UP' if score>.20 else 'DOWN' if score<-.20 else 'SIDEWAYS'
    engine.save(folder/'state');h=pd.DataFrame(engine.history);h.to_parquet(folder/'equity.parquet',index=False)
    pd.DataFrame(engine.book.fills).to_parquet(folder/'fills.parquet',index=False)
    # All raw events retained to support exact accounting audit/restart.
    (folder/'events.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=True)+'\n' for x in engine.book.events),encoding='utf-8')
    hr=np.diff(np.r_[7000.,h.equity])/np.r_[7000.,h.equity.to_numpy()[:-1]]
    def segment(indices):
        r=hr[indices];return dict(days=len(r),net_return=float(np.prod(1+r)-1),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(365)) if len(r)>1 and r.std(ddof=1)>0 else 0.)
    times=pd.to_datetime(h['at'],utc=True)
    windows={'TERRA':['2022-05-07','2022-05-16'],'3AC':['2022-06-10','2022-06-21'],'FTX':['2022-11-06','2022-11-15'],'SVB':['2023-03-10','2023-03-18'],'AUG2024':['2024-08-03','2024-08-09']}
    metric=statistics(engine.history);metric.update(mode=mode,cost=cost,seed=seed,variant=variant,quadrant=quadrant,mapping=mapping,
        evidence='SCENARIO_ONLY',us_survival_verified=False,source_coverage=coverage,external_injections=engine.book.external_injections,
        funding_cash=engine.book.funding_cash,funding_accrued=engine.book.accrued(),costs=engine.book.costs,integer_or_risk_rejections=engine.rejects,
        controller_rejections=len(control.rejections),carry_net_gate_rejections=sum(x['reason']=='CARRY_NET_COST_GATE' for x in control.rejections),
        gross_turnover=sum(abs(x['contracts'])*engine.book.specs[x['asset']].multiplier*x['reference'] for x in engine.book.fills),
        survival_pass=engine.liquidations==0 and engine.hard_breakers<=1,terminal_policy='CONTINUOUS_STATE_NO_FORCED_CLOSE',
        sleeve_pnl={k:sum(p.realized+p.quantity*engine.book.specs[key.split('|')[1]].multiplier*(engine.book.mark.get(key.split('|')[1],p.average)-p.average)+p.funding-p.fees for key,p in engine.book.virtual.items() if key.startswith(k+'|')) for k in CFG['budgets']},
        direction_attribution=direction_attribution(engine.book),
        regimes={regime:segment(h.index[h.market_regime==regime]) for regime in sorted(h.market_regime.unique())},
        stress_windows={name:segment(h.index[(times>=pd.Timestamp(a,tz='UTC'))&(times<pd.Timestamp(b,tz='UTC'))]) for name,(a,b) in windows.items()},
        yearly={str(y):statistics(g.to_dict('records'),float(h.loc[g.index.min()-1,'equity']) if g.index.min()>0 else 7000.) for y,g in h.groupby(pd.to_datetime(h["at"],utc=True).dt.year)})
    from quant_platform.research.governance import write_json
    write_json(folder/'metrics.json',metric)
    write_json(folder/'controller.json',dict(rejections=control.rejections,covariance=control.covariance,residual_delta=control.residual_delta))
    return metric

