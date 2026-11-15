"""Three research sleeves, same frozen risk/cost policy for rules and SIDGTN."""
import math
import numpy as np
import pandas as pd

def trend_features(bars,cfg):
    f=bars.sort_values('candle_ts').set_index('candle_ts')
    f=f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='D'));f.index.name='candle_ts'
    close=f.close;r=close.pct_change(fill_method=None)
    momentum=pd.concat([np.sign(close/close.shift(w)-1) for w in cfg['trend']['momentum']],axis=1).mean(axis=1,skipna=False)
    averages=pd.concat([np.sign(close/close.rolling(w).mean()-1) for w in cfg['trend']['moving_average']],axis=1).mean(axis=1,skipna=False)
    channels=[]
    for w in cfg['trend']['donchian']:
        high=f.high.shift(1).rolling(w).max();low=f.low.shift(1).rolling(w).min()
        trigger=pd.Series(np.nan,index=f.index);trigger[close>high]=1;trigger[close<low]=-1
        state=trigger.ffill().fillna(0.)
        state=state.where(close.notna().rolling(w+1).sum()==w+1)
        channels.append(state)
    breakout=pd.concat(channels,axis=1).mean(axis=1,skipna=False)
    f['raw_score']=(momentum+averages+breakout)/3
    f['ma200']=close.rolling(200).mean()
    blend=(r.rolling(7).std()+r.rolling(28).std())*.5*np.sqrt(365)
    ewma=r.ewm(span=28,adjust=False,min_periods=28).std()*np.sqrt(365)
    f['annual_vol']=pd.concat([blend,ewma],axis=1).max(axis=1).clip(lower=cfg['vol_floor'])
    f['daily_return']=r;f['daily_vol']=f.annual_vol/np.sqrt(365)
    f['valid']=close.notna().rolling(201).sum().eq(201)&f.raw_score.notna()&f.annual_vol.notna()
    f['available_at']=f.index+pd.Timedelta(days=1,hours=cfg['historical_receipt_assumption_hours'])
    f['execution_at']=f.index+pd.Timedelta(days=2)
    return f.reset_index()

def carry_net_expected(annual_rate,days,spot_bps,perp_bps,quantity,spot_price,fixed_fee,contracts):
    if not math.isfinite(annual_rate) or days>28 or days<=0 or quantity<=0:return float('-inf')
    notional=quantity*spot_price
    perp_one_side=max(abs(contracts)*fixed_fee,notional*perp_bps/10000)
    cost=2*notional*spot_bps/10000+2*perp_one_side
    # Stable-basis proxy assumption; no invented convergence gain or limit fill.
    return notional*annual_rate*days/365-cost

def contract_target(score,annual_vol,equity,price,spec,budget,cfg,crowding=1.,event=1.):
    if equity<=0 or annual_vol<=0 or not all(math.isfinite(v) for v in [score,annual_vol,price]):return 0
    daily=max(cfg['vol_floor'],annual_vol)/np.sqrt(365)
    stress=max(cfg['stress_floors'].get(spec.asset,.20),6*daily)
    cap=(1-cfg['maintenance_fraction_assumption'])/stress
    weight=math.copysign(min(cfg['gross_limit'],abs(score)*crowding*event*budget*cfg['annual_vol_scale']/max(annual_vol,cfg['vol_floor']),cap),score) if score else 0.
    raw=weight*equity/(spec.multiplier*price)
    n=int(np.rint(raw))
    # Nearest integer first; if it crosses a bound, shrink toward zero.
    max_n=math.floor(min(cfg['gross_limit'],cap)*equity/(spec.multiplier*price))
    n=max(-max_n,min(max_n,n));return max(-spec.max_contracts,min(spec.max_contracts,n))

class Controller:
    def __init__(self,cfg,mode='filtered',cost='BASE',tactical_only=False):
        self.cfg=cfg;self.mode=mode;self.cost=cfg['costs'][cost];self.tactical_only=tactical_only
        self.rejections=[];self.covariance=[];self.residual_delta=[]

    def plan(self,rows,prices,book,at,directions=None,volatilities=None,covariance=None):
        cfg=self.cfg;eq=book.equity(prices);targets={};spot_targets={};vols={};transfer_assets=[]
        for asset,row in rows.items():
            spec=book.specs[asset];old_carry=book.virtual_position('carry',asset)
            for sleeve in ['trend','carry','tactical']:targets[(sleeve,asset)]=0
            if not bool(row.get('valid',False)):
                self.rejections.append(dict(at=str(at),asset=asset,reason='MISSING_TRAILING_WINDOW'));spot_targets[asset]=0.;continue
            if pd.Timestamp(row['available_at'])>=pd.Timestamp(at):raise ValueError('signal has not arrived before executable open')
            vol=max(cfg['vol_floor'],float((volatilities or {}).get(asset,row['annual_vol'])))
            score=float((directions or {}).get(asset,row['raw_score']))
            if self.mode=='buy_hold':score=1.
            elif self.mode=='long_only':score=max(0.,score)
            elif score<0:score*=cfg['short_multiplier'];score=score if row['close']<row['ma200'] else 0.
            if self.tactical_only or self.mode=='carry_only':score=0.
            crowd=.5 if self.mode not in {'buy_hold','long_only','asymmetric'} and score>0 and row.get('funding_rank30',0)>.90 else 1.
            event=float(row.get('event_coefficient',1.))
            if self.mode not in {'buy_hold','long_only','asymmetric'} and score<0 and row.get('funding_negative3_price_up',False):score=0.
            targets[('trend',asset)]=contract_target(score,vol,eq,prices[asset],spec,cfg['budgets']['trend'],cfg,crowd,event)
            vols[('trend',asset)]=vol/np.sqrt(365)
            ann=float(row.get('funding_mean7_annual',float('nan')))
            carry_mode=self.mode in {'carry','carry_only','full','model'} and cfg['carry']['enabled'] and not self.tactical_only
            carry_n=contract_target(-1.,vol,eq,prices[asset],spec,cfg['budgets']['carry'],cfg) if carry_mode else 0
            carry_n=-min(abs(carry_n),math.floor(cfg['carry']['spot_cap']*max(eq,0)/(len(rows)*spec.multiplier*prices[asset])))
            if carry_n:
                expected=carry_net_expected(ann,cfg['carry']['expected_days'],self.cost['spot_bps'],self.cost['perp_bps'],abs(carry_n)*spec.multiplier,prices[asset],spec.fixed_fee,carry_n)
                if not (ann>=cfg['carry']['entry_annual'] and expected>0):
                    self.rejections.append(dict(at=str(at),asset=asset,reason='CARRY_NET_COST_GATE',annual=None if not math.isfinite(ann) else ann,net_expected=None if not math.isfinite(expected) else expected));carry_n=0
            if old_carry.quantity:
                age=(pd.Timestamp(at)-pd.Timestamp(old_carry.entered_at)).days
                if math.isfinite(ann) and ann>=cfg['carry']['exit_annual'] and ann>0 and age<28:carry_n=old_carry.quantity
                else:carry_n=0
                if carry_n and prices[asset]>=old_carry.average*1.10:
                    carry_n=math.trunc(carry_n/1.10);transfer_assets.append(asset)
            targets[('carry',asset)]=carry_n;spot_targets[asset]=abs(carry_n)*spec.multiplier
            # Tactical uses only then-known funding retreat. Main config remains off.
            enabled=(cfg['tactical']['enabled'] or self.tactical_only)
            high=float(row.get('funding_prior_high',float('nan')));current=float(row.get('funding_current',float('nan')))
            drop=float(row['daily_return']) < -cfg['tactical']['drop_multiple']*vol/np.sqrt(365)
            retreat=math.isfinite(high) and high>0 and math.isfinite(current) and current<=high*(1-cfg['tactical']['funding_fall_fraction'])
            high_rank=float(row.get('funding_prior_rank',0))>=cfg['tactical']['funding_high_rank']
            tactical=book.virtual_position('tactical',asset)
            if enabled and drop and retreat and high_rank:
                ts=max(0.,float((directions or {}).get(asset,1.)))
                targets[('tactical',asset)]=contract_target(ts,vol,eq,prices[asset],spec,cfg['budgets']['tactical'],cfg,event=event)
            elif enabled and tactical.quantity:
                age=(pd.Timestamp(at)-pd.Timestamp(tactical.entered_at)).days
                rebound=prices[asset]>=tactical.average*(1+cfg['tactical']['rebound_multiple']*tactical.daily_vol)
                if age<3 and not rebound:targets[('tactical',asset)]=tactical.quantity
            vols[('tactical',asset)]=vol/np.sqrt(365)
            self.residual_delta.append(dict(at=str(at),asset=asset,quantity=spot_targets[asset]+carry_n*spec.multiplier))
        if covariance is not None and len(rows)==len(covariance):
            assets=list(rows);w=np.array([sum(n for (s,a),n in targets.items() if a==asset and s!='carry')*book.specs[asset].multiplier*prices[asset]/max(eq,1e-9) for asset in assets])
            realized=float(np.sqrt(max(0.,w@np.asarray(covariance)@w)))
            scale=min(1.,cfg['annual_vol_scale']/max(realized,1e-9))
            if scale<1:
                for k in targets:
                    if k[0]!='carry':targets[k]=math.trunc(targets[k]*scale)
            self.covariance.append(dict(at=str(at),unrounded_estimated_vol=realized,scale=scale,note='forecast covariance check, not a guarantee of realized25%'))
        return dict(targets=targets,spot_targets=spot_targets,daily_vol=vols,transfer_assets=transfer_assets)
