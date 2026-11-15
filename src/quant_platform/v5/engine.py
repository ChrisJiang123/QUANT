"""Conservative daily event replay over the V5 single account book.

Intraday points are scenario paths, never reconstructed tick observations.
Primary ambiguous paths check liquidation before assuming a stop can execute.
"""
from dataclasses import replace
import math
import pandas as pd
from .book import Book
from .instruments import cash_settlement_at

class Engine:
    def __init__(self,book,config):
        self.book=book;self.cfg=config;self.peak=book.initial;self.hard_locked=False
        self.liquidations=0;self.hard_breakers=0;self.stops=0;self.rejects=0
        self.cooldown={};self.budget_until={};self.sleeve_peak={};self.history=[]
        self.covariance_checks=[];self.last_day=None;self.gaps=[];self.terminal_liquidated=False

    def maintenance_breach(self,prices):
        return any(p.quantity for p in self.book.physical.values()) and self.book.margin_equity(prices)<self.book.margin(prices,maintenance=True)-1e-9

    def close_all(self,prices,at,reason,liquidation=False,spot=False):
        b=self.book
        b.sync({tuple(k.split('|')):0 for k in b.virtual},prices,at,reason,
               extra_slippage=self.cfg.get('liquidation_extra_slippage',.005) if liquidation else 0.)
        if liquidation:
            fee=self.cfg.get('liquidation_fee_assumption',10.)
            b.derivatives_cash-=fee;b.costs['liquidation']+=fee;self.liquidations+=1
            # Account fee attributed without altering any price PnL.
            active=list(b.virtual.values())
            if active:
                for p in active:p.fees+=fee/len(active)
            b._event('ACCOUNT_LIQUIDATED',at,fee=fee,remaining_equity=b.equity(prices),debt=max(0.,-b.equity(prices)),wallet_debt=max(0.,-b.margin_equity(prices)))
        if spot:
            for a,q in list(b.spot.items()):
                if q:b.spot_target(a,0.,prices[a],at,reason)

    def risk_at(self,prices,at):
        b=self.book;b.mark_to(prices);equity=b.equity(prices)
        self.peak=max(self.peak,equity);dd=1-equity/self.peak
        if self.maintenance_breach(prices):
            self.close_all(prices,at,'liquidation',True);return 'LIQUIDATED'
        if dd>=self.cfg.get('hard_drawdown',.25) and not self.hard_locked:
            self.hard_locked=True;self.hard_breakers+=1
            self.close_all(prices,at,'hard_breaker',spot=True)
            b._event('HARD_BREAKER_LOCKED',at,drawdown=dd);return 'HARD_LOCKED'
        if equity<=0:return 'INSOLVENT'
        return 'SOFT' if dd>=self.cfg.get('soft_drawdown',.15) else 'NORMAL'

    def restart(self,at,reason):
        if not reason:raise ValueError('explicit restart justification required')
        self.book._event('EXPLICIT_RESEARCH_RESTART',at,reason=reason)
        self.hard_locked=False;self.peak=self.book.equity()

    def constrain(self,targets,prices,at,force=False):
        b=self.book;eq=b.equity(prices)
        if self.hard_locked or eq<=0:return {k:0 for k in targets}
        proposed={}
        soft=(1-eq/self.peak)>=self.cfg.get('soft_drawdown',.15)
        for (s,a),n in targets.items():
            n=b.specs[a].validate_quantity(n);old=b.virtual_position(s,a).quantity
            if self.cooldown.get((s,a),pd.Timestamp.min.tz_localize('UTC'))>pd.Timestamp(at):n=0
            if soft:n=math.trunc(n*.5)
            if self.budget_until.get(s):n=math.trunc(n*.5)
            # Band never blocks reductions, expiry, reversals or risk actions.
            if not force and not soft and not self.budget_until.get(s) and n*old>0 and abs(n-old)<.20*abs(old):n=old
            proposed[(s,a)]=n
        def limits(t):
            gross=sum(abs(n)*b.specs[a].multiplier*prices[a] for (s,a),n in t.items())+sum(q*prices[a] for a,q in b.spot.items())
            margin=sum(b.specs[a].initial_margin(n,prices[a]) for (s,a),n in t.items())
            net={a:sum(n for (s,x),n in t.items() if x==a) for a in b.specs}
            physical=sum(b.specs[a].initial_margin(n,prices[a]) for a,n in net.items())
            return gross,margin,physical
        gross,margin,physical=limits(proposed)
        cap=min(1.,self.cfg.get('gross_limit',10)*eq/max(gross,1e-9),
                self.cfg.get('margin_limit',.5)*eq/max(margin,1e-9),
                max(0.,b.margin_equity(prices))/max(physical,1e-9))
        if cap<1:proposed={k:math.trunc(n*cap) for k,n in proposed.items()}
        directional={a:sum(n for (s,x),n in proposed.items() if x==a and s!='carry') for a in b.specs}
        for a,n in directional.items():
            if not n:continue
            daily=max((p.daily_vol for k,p in b.virtual.items() if k.endswith('|'+a)),default=.10/math.sqrt(365))
            pressure=max(self.cfg.get('stress_floors',{}).get(a,.20),6*daily)
            maximum=math.floor((1-self.cfg.get('maintenance_fraction_assumption',.5))/pressure*eq/(b.specs[a].multiplier*prices[a]))
            if abs(n)>maximum:
                excess=abs(n)-maximum
                for sleeve in ['tactical','trend']:
                    key=(sleeve,a);value=proposed.get(key,0)
                    if value*n>0:
                        reduction=min(abs(value),excess);proposed[key]=value-int(math.copysign(reduction,value));excess-=reduction
        actual_notional=sum(b.specs[a].notional(p.quantity,prices[a]) for a,p in b.physical.items() if p.quantity)
        if actual_notional:
            distance=(b.margin_equity(prices)-b.margin(prices,maintenance=True))/actual_notional
            daily=max((p.daily_vol for p in b.virtual.values() if p.quantity),default=.02)
            if distance<2*daily:
                for key in proposed:
                    old=b.virtual_position(*key).quantity
                    proposed[key]=int(math.copysign(min(abs(proposed[key]),abs(old)//2),old)) if proposed[key]*old>0 else 0
                b._event('LIQUIDATION_DISTANCE_REDUCTION',at,distance=distance,threshold=2*daily)
        for k,n in targets.items():
            if n and not proposed[k]:self.rejects+=1
        assert limits(proposed)[0]<=self.cfg.get('gross_limit',10)*eq+1e-7
        assert limits(proposed)[1]<=self.cfg.get('margin_limit',.5)*eq+1e-7
        return proposed

    def margin_update(self,asset,initial_fraction,maintenance_fraction,prices,at):
        if not 0<maintenance_fraction<=1 or not 0<initial_fraction<=1:raise ValueError('invalid margin scenario')
        b=self.book;b.specs[asset]=replace(b.specs[asset],initial_margin_fraction=initial_fraction,maintenance_fraction_of_initial=maintenance_fraction)
        b._event('MARGIN_RULE_CHANGE',at,asset=asset,initial_fraction=initial_fraction,maintenance_fraction=maintenance_fraction)
        # A rate jump may already liquidate; do not pre-emptively rescue with future money.
        if self.maintenance_breach(prices):self.close_all(prices,at,'margin_hike_liquidation',True);return
        eq=b.equity(prices);margin=b.margin(prices,virtual=True)
        if margin>eq*self.cfg.get('margin_limit',.5):
            scale=max(0.,eq)*self.cfg.get('margin_reduce_target',.399)/max(margin,1e-9)
            b.sync({tuple(k.split('|')):math.trunc(p.quantity*scale) for k,p in b.virtual.items()},prices,at,'margin_hike_reduction')

    def stop_at(self,prices,adverse,at):
        b=self.book
        for key,p in list(b.virtual.items()):
            if not p.quantity:continue
            s,a=key.split('|');direction=1 if p.quantity>0 else -1
            # Carry hedges have their own two-leg controller, not a directional stop.
            if s=='carry':continue
            stop=p.average*(1-direction*self.cfg.get('stop_daily_vol_multiple',2.5)*p.daily_vol)
            hit=adverse[a]<=stop if direction>0 else adverse[a]>=stop
            if hit:
                fill=min(prices[a],stop) if direction>0 else max(prices[a],stop)
                execution={**prices,a:max(b.specs[a].tick,fill)}
                b.sync({(s,a):0},execution,at,'stop_gap_or_adverse_path')
                self.cooldown[(s,a)]=pd.Timestamp(at)+pd.Timedelta(days=self.cfg.get('cooldown_days',3))
                self.stops+=1

    def step(self,at,bars,targets=None,daily_vol=None,funding_events=None,margin_changes=None,
             evidence='SCENARIO_ONLY',favorable_first=False,spot_targets=None,transfer_assets=None,trading_halted=False,maintenance_window=False):
        at=pd.Timestamp(at)
        if at.tzinfo is None or at.utcoffset().total_seconds()!=0:raise ValueError('UTC event time required')
        if self.last_day is not None and at<=pd.Timestamp(self.last_day):raise ValueError('day already consumed; resume from next day')
        b=self.book
        held={a for a,p in b.physical.items() if p.quantity}|{a for a,q in b.spot.items() if q}
        if not held.issubset(bars) or any(not all(math.isfinite(float(r[k])) and float(r[k])>0 for k in ['open','high','low','close']) for r in bars.values()):
            self.gaps.append(str(at));b._event('DATA_GAP_NO_FILL',at,assets=sorted(held-set(bars)))
            self.last_day=str(at);return
        opens={a:float(r['open']) for a,r in bars.items()}
        # Current executable bar must exist even for an unheld target asset.
        missing_targets={a for s,a in (targets or {}) if a not in bars}
        if missing_targets:self.gaps.append(str(at));targets=None
        b.mark_to(opens);b.receive_transfers(at);b.settle_funding(at)
        if trading_halted or maintenance_window:
            b._event('TRADING_HALTED' if trading_halted else 'MAINTENANCE_WINDOW',at,scheduled=maintenance_window)
            targets=None
        initial_risk=self.risk_at(opens,at)
        if margin_changes:
            for a,initial,maintenance in margin_changes:self.margin_update(a,initial,maintenance,opens,at)
        if targets is not None and initial_risk not in {'LIQUIDATED','INSOLVENT'}:
            for key,p in b.virtual.items():targets.setdefault(tuple(key.split('|')),p.quantity)
            for key,p in b.virtual.items():
                sleeve,a=key.split('|');maximum=self.cfg.get('max_holding_days',{}).get(sleeve,28)
                if p.entered_at and at-pd.Timestamp(p.entered_at)>=pd.Timedelta(days=maximum):targets[(sleeve,a)]=0
            constrained=self.constrain(targets,opens,at)
            for a in spot_targets or {}:
                if ('carry',a) not in constrained:continue
                quantity=abs(constrained[('carry',a)])*b.specs[a].multiplier
                old=b.spot[a];delta=quantity-old
                if delta>0 and delta*opens[a]*(1+b.spot_bps/10000)>b.spot_cash+1e-8:
                    constrained[('carry',a)]=b.virtual_position('carry',a).quantity
                    self.rejects+=1;quantity=old
                if quantity<old:
                    before=b.spot_cash;b.spot_target(a,quantity,opens[a],at,'carry_reduce_or_exit')
                    if a in (transfer_assets or []) and b.spot_cash>before:
                        b.transfer(b.spot_cash-before,at,delay_hours=self.cfg.get('carry',{}).get('transfer_delay_hours',24),fee=self.cfg.get('carry',{}).get('transfer_fee',0.))
            # Recheck immediately available cash after explicit fee/slippage reserve.
            b.sync(constrained,opens,at,'rebalance',daily_vol=daily_vol)
            for a in spot_targets or {}:
                quantity=abs(b.virtual_position('carry',a).quantity)*b.specs[a].multiplier
                if quantity>b.spot[a]:
                    if not b.spot_target(a,quantity,opens[a],at,'carry_open_costed_taker'):
                        b.sync({('carry',a):0},opens,at,'single_leg_failure_unwind')
                b._event('CARRY_DELTA_CHECK',at,asset=a,residual_quantity=b.spot[a]+b.virtual_position('carry',a).quantity*b.specs[a].multiplier)
            self.risk_at(opens,at)
        # Funding intervals use actual held time. The end-of-day rate is never a signal.
        timeline={}
        end=at+pd.Timedelta(days=1)
        for event in funding_events or []:
            t=pd.Timestamp(event['event_at'])
            if not at<=t<end:raise ValueError('funding event outside bar')
            timeline.setdefault(t,[]).append(event)
            due=pd.Timestamp(cash_settlement_at(t.tz_convert('America/Chicago').date()))
            if due<t:due=pd.Timestamp(cash_settlement_at((t.tz_convert('America/Chicago')+pd.Timedelta(days=1)).date()))
            if at<=due<end:timeline.setdefault(due,[])
        for x in b.pending_funding+b.transfers:
            due=pd.Timestamp(x['due'])
            if at<=due<end:timeline.setdefault(due,[])
        for t,events in sorted(timeline.items()):
            for event in events:b.accrue_funding(event['asset'],event['rate'],event['price'],t,event['event_id'],
                    stress=event.get('stress',False),interval_start=event.get('interval_start'))
            b.settle_funding(t);b.receive_transfers(t);self.risk_at(opens,t)
        # Any remaining broker settlement occurs before daily extreme in this
        # conservative daily scenario. This is not claimed intraday mark history.
        ct=at+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)
        b.settle_funding(ct)
        adverse={a:float(r['low'] if b.physical[a].quantity>=0 else r['high']) for a,r in bars.items()}
        favorable={a:float(r['high'] if b.physical[a].quantity>=0 else r['low']) for a,r in bars.items()}
        ambiguous=self.maintenance_breach(adverse)
        if favorable_first:
            self.risk_at(favorable,ct);self.stop_at(opens,adverse,ct)
        if ambiguous and any(p.quantity for p in b.physical.values()):
            if self.maintenance_breach(opens):cross=opens
            else:
                lo=0.;hi=1.
                for _ in range(45):
                    mid=(lo+hi)/2;prices={a:opens[a]+mid*(adverse[a]-opens[a]) for a in opens}
                    if self.maintenance_breach(prices):hi=mid
                    else:lo=mid
                cross={a:opens[a]+hi*(adverse[a]-opens[a]) for a in opens}
            self.close_all(cross,ct,'ambiguous_daily_liquidation_before_stop',True)
        elif not favorable_first:self.stop_at(opens,adverse,ct)
        self.risk_at(adverse,ct);self.risk_at(favorable,ct)
        closes={a:float(r['close']) for a,r in bars.items()}
        self.risk_at(closes,ct);b.assert_conservation(closes)
        equity=b.equity(closes)
        for sleeve,budget in self.cfg.get('budgets',{}).items():
            pnl=sum(p.realized+p.quantity*b.specs[k.split('|')[1]].multiplier*(closes.get(k.split('|')[1],p.average)-p.average)+p.funding-p.fees for k,p in b.virtual.items() if k.startswith(sleeve+'|'))
            if sleeve=='carry':pnl+=b.spot_realized+sum(q*(closes[a]-b.spot_average[a]) for a,q in b.spot.items() if q)-b.costs['spot']
            value=b.initial*budget+pnl;peak=self.sleeve_peak.get(sleeve,b.initial*budget)
            if sleeve in self.budget_until and pd.Timestamp(ct)>=pd.Timestamp(self.budget_until[sleeve]) and value>=peak:
                self.budget_until.pop(sleeve);b._event('SLEEVE_BUDGET_RESTORED',ct,sleeve=sleeve)
            if value<peak*(1-2*budget*self.cfg.get('annual_vol_scale',.25)) and sleeve not in self.budget_until:
                self.budget_until[sleeve]=str(ct+pd.Timedelta(days=30));b._event('SLEEVE_BUDGET_HALVED',ct,sleeve=sleeve,until=self.budget_until[sleeve])
            self.sleeve_peak[sleeve]=max(value,peak)
        self.history.append(dict(at=str(at),equity=equity,derivatives_cash=b.derivatives_cash,spot_cash=b.spot_cash,
            accrued=b.accrued(),gross=b.gross(closes)/max(abs(equity),1e-9),
            net=sum(p.quantity*b.specs[a].multiplier*closes[a] for a,p in b.physical.items() if p.quantity)/max(abs(equity),1e-9),
            initial_margin=b.margin(closes),virtual_initial_margin=b.margin(closes,virtual=True),
            funding_cash=b.funding_cash,stops=self.stops,liquidations=self.liquidations,hard_breakers=self.hard_breakers,
            hard_locked=self.hard_locked,evidence=evidence))
        self.last_day=str(at)

    def save(self,folder):
        from pathlib import Path
        from quant_platform.research.governance import write_json
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);self.book.save(folder/'book.json')
        state={k:v for k,v in vars(self).items() if k not in {'book','cooldown'}}
        state['cooldown']={'|'.join(k):str(v) for k,v in self.cooldown.items()};write_json(folder/'engine.json',state)

    @classmethod
    def load(cls,folder):
        from pathlib import Path
        import json
        folder=Path(folder);d=json.loads((folder/'engine.json').read_text(encoding='utf-8'))
        e=cls(Book.load(folder/'book.json'),d['cfg']);d['cooldown']={tuple(k.split('|')):pd.Timestamp(v) for k,v in d['cooldown'].items()}
        e.__dict__.update(d);return e
