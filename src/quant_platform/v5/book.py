"""Single V5 offline account ledger. Virtual attribution never replaces net cash.

There is no order API. Contract fills, spot fills and internal transfers are
explicit simulation events. Account equity is allowed to remain negative.
"""
from dataclasses import asdict,dataclass,field
from datetime import datetime,timedelta,timezone
import json,math
import pandas as pd
from quant_platform.research.governance import digest
from .instruments import InstrumentSpec,cash_settlement_at

@dataclass
class Position:
    quantity:int=0
    average:float=0.
    realized:float=0.
    fees:float=0.
    execution_cost:float=0.
    funding:float=0.
    entered_at:str|None=None
    daily_vol:float=.02

    def transact(self,target,price,at):
        old=self.quantity; delta=target-old
        if not delta:return 0.
        pnl=0.
        if old and old*delta<0:
            closed=min(abs(old),abs(delta))
            pnl=closed*(price-self.average)*(1 if old>0 else -1)
        if not old or old*target<0:self.average=price
        elif old*delta>0:self.average=(abs(old)*self.average+abs(delta)*price)/abs(target)
        if not target:self.average=0.;self.entered_at=None
        elif not old or old*target<0:self.entered_at=str(at)
        self.quantity=target;self.realized+=pnl
        return pnl

class Book:
    def __init__(self,specs,initial=7000.,perp_bps=5.,spot_bps=85.,wallet_fraction=.5):
        self.specs=specs;self.initial=initial
        self.derivatives_cash=initial*wallet_fraction;self.spot_cash=initial*(1-wallet_fraction)
        self.perp_bps=perp_bps;self.spot_bps=spot_bps
        self.physical={a:Position() for a in specs};self.virtual={}
        self.spot={a:0. for a in specs};self.spot_average={a:0. for a in specs}
        self.spot_realized=0.;self.pending_funding=[];self.transfers=[]
        self.seen_funding=set();self.fills=[];self.events=[]
        self.costs=dict(fixed=0.,execution=0.,spot=0.,liquidation=0.,transfer=0.)
        self.funding_cash=0.;self.last_at=None;self.mark={};self.external_injections=0.;self.exposure_history={}

    def _event(self,kind,at,**values):
        if self.last_at is not None and pd.Timestamp(at)<pd.Timestamp(self.last_at):
            raise ValueError('account events must be chronological')
        self.last_at=str(at)
        e=dict(kind=kind,at=str(at),previous_hash=self.events[-1]['hash'] if self.events else '0'*64,**values)
        e['hash']=digest(e);self.events.append(e);return e

    def virtual_position(self,sleeve,asset):return self.virtual.setdefault(sleeve+'|'+asset,Position())
    def mark_to(self,prices):
        for a,p in prices.items():
            if a not in self.specs or not math.isfinite(float(p)) or p<=0:raise ValueError('positive finite mark required')
            self.mark[a]=float(p)

    def unrealized(self,prices=None):
        prices=prices or self.mark
        return sum(p.quantity*self.specs[a].multiplier*(prices[a]-p.average) for a,p in self.physical.items() if p.quantity)

    def accrued(self):return sum(e['amount'] for e in self.pending_funding)
    def margin_equity(self,prices=None):return self.derivatives_cash+self.unrealized(prices)
    def equity(self,prices=None):
        prices=prices or self.mark
        spot=sum(q*prices[a] for a,q in self.spot.items() if q)
        return self.margin_equity(prices)+self.spot_cash+spot+self.accrued()+sum(t['amount'] for t in self.transfers)

    def margin(self,prices=None,maintenance=False,virtual=False):
        prices=prices or self.mark
        positions=((key.split('|')[1],p) for key,p in self.virtual.items()) if virtual else self.physical.items()
        return sum((self.specs[a].maintenance if maintenance else self.specs[a].initial_margin)(p.quantity,prices[a]) for a,p in positions if p.quantity)

    def gross(self,prices=None):
        prices=prices or self.mark
        return sum(abs(p.quantity)*self.specs[k.split('|')[1]].multiplier*prices[k.split('|')[1]] for k,p in self.virtual.items() if p.quantity)+sum(abs(q)*prices[a] for a,q in self.spot.items() if q)

    def _record_exposure(self,key,delta,at):
        if not delta:return
        entries=self.exposure_history.setdefault(key,[]);t=pd.Timestamp(at).timestamp()
        old=entries[-1] if entries else [t,0.,0.]
        entries.append([t,old[1]+old[2]*(t-old[0]),old[2]+delta])

    def sync(self,targets,prices,at,reason='target',daily_vol=None,extra_slippage=0.):
        """Targets are virtual integer contracts. Fees apply only to net venue fills."""
        self.mark_to(prices)
        for (sleeve,a),n in targets.items():self.specs[a].validate_quantity(n)
        assets=sorted({a for _,a in targets})
        for a in assets:
            changes={s:int(n)-self.virtual_position(s,a).quantity for (s,asset),n in targets.items() if asset==a}
            delta=sum(changes.values());spec=self.specs[a];reference=float(prices[a])
            units=abs(delta)*spec.multiplier;fixed=abs(delta)*spec.fixed_fee
            expected=units*reference*self.perp_bps/10000
            execution=max(expected-fixed,0.)+units*reference*extra_slippage
            fill=reference+(1 if delta>0 else -1)*execution/units if delta else reference
            physical=self.physical[a]
            self._record_exposure(a,delta,at)
            raw=physical.transact(physical.quantity+delta,fill,at);realized=raw*spec.multiplier
            # Position.transact records price-unit PnL; convert its counter to USD.
            physical.realized+=raw*(spec.multiplier-1)
            self.derivatives_cash+=realized-fixed
            physical.fees+=fixed;physical.execution_cost+=execution
            total_change=sum(abs(v) for v in changes.values())
            for s,change in changes.items():
                p=self.virtual_position(s,a)
                self._record_exposure(s+'|'+a,change,at)
                raw=p.transact(p.quantity+change,fill,at);p.realized+=raw*(spec.multiplier-1)
                p.fees+=fixed*abs(change)/total_change if total_change else 0.
                p.execution_cost+=execution*abs(change)/total_change if total_change else 0.
                if daily_vol and (s,a) in daily_vol:p.daily_vol=float(daily_vol[(s,a)])
            self.costs['fixed']+=fixed;self.costs['execution']+=execution
            if delta:
                self.fills.append(dict(at=str(at),asset=a,contracts=delta,price=fill,reference=reference,
                    fixed_fee=fixed,execution_cost=execution,reason=reason,realized=realized))
            self._event('NET_FILL' if delta else 'INTERNAL_ATTRIBUTION',at,asset=a,net_contracts=delta,
                virtual_changes=changes,price=fill,fixed=fixed,execution=execution,reason=reason)
        self.assert_conservation(prices)

    def spot_target(self,asset,quantity,price,at,reason='carry'):
        if not math.isfinite(quantity) or quantity<0:raise ValueError('unlevered long spot only')
        old=self.spot[asset];delta=quantity-old;fee=abs(delta)*price*self.spot_bps/10000
        if delta>0 and delta*price+fee>self.spot_cash+1e-8:
            self._event('SPOT_REJECTED_CASH',at,asset=asset,quantity=quantity);return False
        self.spot_cash-=delta*price+fee
        if delta>0:self.spot_average[asset]=(old*self.spot_average[asset]+delta*price)/quantity
        else:self.spot_realized+=-delta*(price-self.spot_average[asset])
        self.spot[asset]=quantity;self.costs['spot']+=fee
        self._event('SPOT_FILL',at,asset=asset,quantity=delta,price=price,fee=fee,reason=reason)
        return True

    def transfer(self,amount,at,delay_hours=24,fee=0.):
        if amount<=0 or fee<0 or delay_hours<0:raise ValueError('valid internal transfer required')
        if amount+fee>self.spot_cash+1e-8:raise ValueError('no external injection or unrealized spot collateral')
        self.spot_cash-=amount+fee;self.costs['transfer']+=fee
        due=pd.Timestamp(at)+pd.Timedelta(hours=delay_hours)
        self.transfers.append(dict(amount=float(amount),due=str(due)))
        self._event('TRANSFER_REQUESTED',at,amount=amount,due=str(due),fee=fee)

    def receive_transfers(self,at):
        due=[x for x in self.transfers if pd.Timestamp(x['due'])<=pd.Timestamp(at)]
        for t in due:self.derivatives_cash+=t['amount'];self._event('TRANSFER_RECEIVED',at,**t)
        self.transfers=[x for x in self.transfers if x not in due]

    def exposure_during(self,asset,start,end,sleeve=None):
        start=pd.Timestamp(start);end=pd.Timestamp(end)
        if end<=start:raise ValueError('positive funding interval required')
        from bisect import bisect_right
        entries=self.exposure_history.get(asset if sleeve is None else sleeve+'|'+asset,[])
        def integral(t):
            i=bisect_right(entries,[t,float('inf'),float('inf')])-1
            if i<0:return 0.
            stamp,value,quantity=entries[i];return value+quantity*(t-stamp)
        return (integral(end.timestamp())-integral(start.timestamp()))/(end-start).total_seconds()

    def accrue_funding(self,asset,rate,price,event_at,event_id,stress=False,interval_start=None):
        if event_id in self.seen_funding:return False
        if not math.isfinite(rate):raise ValueError('missing funding is not zero')
        q=self.physical[asset].quantity if interval_start is None else self.exposure_during(asset,interval_start,event_at)
        spec=self.specs[asset]
        effective=rate
        if stress and q:effective*=1.5 if q*rate>0 else .5
        amount=-q*spec.multiplier*price*effective
        stamp=pd.Timestamp(event_at);due=pd.Timestamp(cash_settlement_at(stamp.tz_convert('America/Chicago').date()))
        if due<stamp:due=pd.Timestamp(cash_settlement_at((stamp.tz_convert('America/Chicago')+pd.Timedelta(days=1)).date()))
        self.pending_funding.append(dict(asset=asset,amount=amount,due=str(due),event_id=event_id))
        for key,p in self.virtual.items():
            if key.endswith('|'+asset):
                vq=p.quantity if interval_start is None else self.exposure_during(asset,interval_start,event_at,key.split('|')[0])
                p.funding+=-vq*spec.multiplier*price*effective
        self.seen_funding.add(event_id)
        self._event('FUNDING_ACCRUED',event_at,asset=asset,amount=amount,rate=rate,effective_rate=effective,due=str(due),event_id=event_id)
        return True

    def settle_funding(self,at):
        due=[x for x in self.pending_funding if pd.Timestamp(x['due'])<=pd.Timestamp(at)]
        amount=sum(x['amount'] for x in due)
        self.derivatives_cash+=amount;self.funding_cash+=amount
        self.pending_funding=[x for x in self.pending_funding if x not in due]
        if due:self._event('FUNDING_CASH_SETTLED',at,amount=amount,events=[x['event_id'] for x in due])
        return amount

    def assert_conservation(self,prices=None):
        prices=prices or self.mark
        for a,p in self.physical.items():
            virtual=[v for k,v in self.virtual.items() if k.endswith('|'+a)]
            assert sum(v.quantity for v in virtual)==p.quantity,'virtual/physical quantity diverged'
            mark=prices.get(a,p.average or 1.)
            physical_pnl=p.realized+p.quantity*self.specs[a].multiplier*(mark-p.average)
            virtual_pnl=sum(v.realized+v.quantity*self.specs[a].multiplier*(mark-v.average) for v in virtual)
            assert abs(physical_pnl-virtual_pnl)<1e-6*max(1,abs(physical_pnl)),'attribution PnL diverged'
        total_virtual_funding=sum(p.funding for p in self.virtual.values())
        assert abs(total_virtual_funding-self.funding_cash-self.accrued())<1e-6,'funding double counted'
        assert self.external_injections==0

    def save(self,path):
        from quant_platform.research.governance import write_json
        state={k:v for k,v in vars(self).items() if k not in {'specs','physical','virtual','seen_funding'}}
        state.update(specs={a:asdict(s) for a,s in self.specs.items()},physical={a:asdict(p) for a,p in self.physical.items()},
                     virtual={a:asdict(p) for a,p in self.virtual.items()},seen_funding=sorted(self.seen_funding))
        write_json(path,state)

    @classmethod
    def load(cls,path):
        d=json.loads(__import__('pathlib').Path(path).read_text(encoding='utf-8'))
        obj=cls({a:InstrumentSpec(**s) for a,s in d.pop('specs').items()})
        physical=d.pop('physical');virtual=d.pop('virtual');seen=d.pop('seen_funding');obj.__dict__.update(d)
        obj.physical={a:Position(**p) for a,p in physical.items()};obj.virtual={a:Position(**p) for a,p in virtual.items()};obj.seen_funding=set(seen)
        obj.assert_conservation();return obj
