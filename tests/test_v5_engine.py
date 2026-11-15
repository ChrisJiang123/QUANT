import copy
import math
import numpy as np
import pandas as pd
import pytest
from quant_platform.v5.book import Book
from quant_platform.v5.engine import Engine
from quant_platform.v5.instruments import InstrumentSpec,cash_settlement_at,annualize_rate

T=pd.Timestamp('2020-01-01',tz='UTC')

def spec(mult=1.,fixed=0.,margin=.1):
    return {'X':InstrumentSpec('scenario:X','X',mult,.01,fixed_fee=fixed,initial_margin_fraction=margin)}
def cfg(**kw):
    return dict(gross_limit=10.,margin_limit=1.,margin_reduce_target=.399,soft_drawdown=2.,hard_drawdown=3.,
                stop_daily_vol_multiple=2.5,cooldown_days=3,max_holding_days={'trend':28,'carry':28,'tactical':3},**kw)
def bar(o=100.,lo=100.,hi=100.,c=100.):return {'X':dict(open=o,low=lo,high=hi,close=c,volume=1e8)}

def test_partial_close_reverse_and_linear_pnl():
    b=Book(spec(.5),initial=1000,perp_bps=0,wallet_fraction=1)
    b.sync({('trend','X'):4},{'X':100},T)
    b.sync({('trend','X'):2},{'X':110},T+pd.Timedelta(hours=1))
    assert b.derivatives_cash==1010
    b.sync({('trend','X'):-3},{'X':110},T+pd.Timedelta(hours=2))
    assert b.derivatives_cash==1020
    assert b.equity({'X':120})==1005
    b.assert_conservation({'X':120})

@pytest.mark.parametrize('n',[.5,True,float('nan'),float('inf')])
def test_integer_contract_rejection(n):
    b=Book(spec())
    with pytest.raises(ValueError):b.sync({('trend','X'):n},{'X':100},T)
    assert not b.fills

def test_netting_attribution_random_path_and_no_gross_hiding():
    b=Book(spec(.01,.0113),initial=7000,perp_bps=5)
    rng=np.random.default_rng(123)
    for i in range(70):
        p=float(rng.uniform(70,150))
        b.sync({('trend','X'):int(rng.integers(-20,20)),('carry','X'):int(rng.integers(-20,0))},{'X':p},T+pd.Timedelta(hours=i))
        b.assert_conservation({'X':p})
    b.sync({('trend','X'):10,('carry','X'):-10},{'X':100},T+pd.Timedelta(hours=71))
    assert b.physical['X'].quantity==0 and b.gross({'X':100})==20

def test_fixed_fee_is_not_double_charged_with_bps():
    b=Book(spec(.01,.0113),perp_bps=5)
    b.sync({('trend','X'):10},{'X':10000},T)
    assert b.costs['fixed']==pytest.approx(.113)
    assert b.costs['fixed']+b.costs['execution']==pytest.approx(.5)
    assert 7000-b.equity({'X':10000})==pytest.approx(.5)

def test_funding_held_time_cash_and_accrual_not_double_counted():
    b=Book(spec(),initial=1000,perp_bps=0,wallet_fraction=1)
    b.sync({('trend','X'):4},{'X':100},T+pd.Timedelta(minutes=30))
    b.accrue_funding('X',.01,100,T+pd.Timedelta(hours=1),'h1',interval_start=T)
    assert b.accrued()==-2
    assert b.equity({'X':100})==998 and b.derivatives_cash==1000
    b.settle_funding(T+pd.Timedelta(hours=22))
    assert b.derivatives_cash==998 and b.equity({'X':100})==998
    assert not b.accrue_funding('X',.01,100,T+pd.Timedelta(hours=23),'h1')
    b.assert_conservation({'X':100})

def test_funding_on_already_closed_position_and_net_short_sign():
    b=Book(spec(),initial=1000,perp_bps=0)
    b.sync({('trend','X'):-2},{'X':100},T)
    b.sync({('trend','X'):0},{'X':100},T+pd.Timedelta(minutes=30))
    b.accrue_funding('X',.01,100,T+pd.Timedelta(hours=1),'h1',interval_start=T)
    assert b.accrued()==1
    b.assert_conservation({'X':100})

def test_stress_funding_net_cash_and_virtual_conserve():
    b=Book(spec(),perp_bps=0)
    b.sync({('trend','X'):4,('carry','X'):-2},{'X':100},T)
    b.accrue_funding('X',.01,100,T+pd.Timedelta(hours=1),'s',stress=True)
    assert b.accrued()==-3
    assert sum(x.funding for x in b.virtual.values())==-3

def test_cash_timezone_dst_and_actual_frequency():
    assert cash_settlement_at('2026-01-15').hour==21
    assert cash_settlement_at('2026-07-15').hour==20
    assert annualize_rate(.0001,8)==pytest.approx(.1095)
    assert annualize_rate(.0001,1)==pytest.approx(.876)

def test_wallet_isolation_delayed_transfer_and_no_injection():
    b=Book(spec(),initial=1000,spot_bps=0)
    assert b.spot_target('X',4,100,T)
    b.mark_to({'X':200})
    assert b.margin_equity()==500
    with pytest.raises(ValueError):b.transfer(200,T+pd.Timedelta(hours=1))
    b.spot_target('X',0,200,T+pd.Timedelta(hours=2))
    b.transfer(600,T+pd.Timedelta(hours=3),delay_hours=24)
    assert b.derivatives_cash==500 and b.equity()==1400
    b.receive_transfers(T+pd.Timedelta(hours=26));assert b.derivatives_cash==500
    b.receive_transfers(T+pd.Timedelta(hours=27));assert b.derivatives_cash==1100
    assert b.equity()==1400 and b.external_injections==0

def test_twenty_percent_gap_actually_liquidates_and_retains_debt():
    b=Book(spec(),initial=1000,perp_bps=0,wallet_fraction=1);e=Engine(b,cfg())
    # Deliberately unsafe initial inventory; normal V5 risk sizing would reject it.
    b.sync({('trend','X'):100},{'X':100},T,daily_vol={('trend','X'):.02})
    e.step(T+pd.Timedelta(days=1),bar(80,79,81,80))
    assert e.liquidations==1
    assert b.equity()<0
    assert b.costs['liquidation']==10
    assert any(x['reason']=='liquidation' and x['price']<=80 for x in b.fills)

def test_same_bar_stop_does_not_automatically_rescue_liquidation():
    results=[]
    for favorable in [False,True]:
        b=Book(spec(),initial=1000,perp_bps=0,wallet_fraction=1);e=Engine(b,cfg())
        b.sync({('trend','X'):100},{'X':100},T,daily_vol={('trend','X'):.02})
        e.step(T,bar(lo=80,hi=101),favorable_first=favorable)
        results.append((e.liquidations,e.stops))
    assert results==[(1,0),(0,1)]

def test_funding_cash_can_trigger_liquidation():
    b=Book(spec(),initial=1000,perp_bps=0,wallet_fraction=1);e=Engine(b,cfg())
    b.sync({('trend','X'):100},{'X':100},T)
    b.accrue_funding('X',.06,100,T+pd.Timedelta(hours=1),'large')
    assert not e.maintenance_breach({'X':100})
    b.settle_funding(T+pd.Timedelta(hours=21))
    assert e.maintenance_breach({'X':100})
    assert e.risk_at({'X':100},T+pd.Timedelta(hours=21))=='LIQUIDATED'

def test_margin_hike_waives_band_and_reduces_below_40_percent():
    b=Book(spec(),initial=1000,perp_bps=0,wallet_fraction=1);c=cfg();c['margin_limit']=.5;e=Engine(b,c)
    b.sync({('trend','X'):40},{'X':100},T)
    e.margin_update('X',.2,.5,{'X':100},T+pd.Timedelta(hours=1))
    assert b.margin({'X':100},virtual=True)<400
    assert b.virtual['trend|X'].quantity<40

def test_hard_breaker_locks_and_explicit_restart_only():
    b=Book(spec(),initial=1000,perp_bps=0,wallet_fraction=1);c=cfg();c['hard_drawdown']=.25;e=Engine(b,c)
    b.sync({('trend','X'):10},{'X':100},T)
    e.risk_at({'X':70},T+pd.Timedelta(hours=1))
    assert e.hard_locked and e.hard_breakers==1
    assert e.constrain({('trend','X'):1},{'X':70},T+pd.Timedelta(hours=2))=={('trend','X'):0}
    with pytest.raises(ValueError):e.restart(T+pd.Timedelta(hours=3),'')
    e.restart(T+pd.Timedelta(hours=3),'explicit offline fixture reset');assert not e.hard_locked

def test_save_resume_continuous_account_equals_uninterrupted(tmp_path):
    a=Engine(Book(spec(),initial=1000,perp_bps=5,wallet_fraction=1),cfg())
    a.step(T,bar(),{('trend','X'):2});a.save(tmp_path)
    b=Engine.load(tmp_path)
    for engine in [a,b]:engine.step(T+pd.Timedelta(days=1),bar(101,100,103,102),{('trend','X'):3})
    assert a.book.equity()==pytest.approx(b.book.equity())
    assert a.book.events==b.book.events
    with pytest.raises(ValueError):b.step(T+pd.Timedelta(days=1),bar())

def test_gap_and_missing_funding_are_not_fabricated():
    b=Book(spec(),perp_bps=0);e=Engine(b,cfg())
    e.step(T,bar(),{('trend','X'):2})
    e.step(T+pd.Timedelta(days=1),{})
    assert e.gaps and len(b.fills)==1
    with pytest.raises(ValueError):b.accrue_funding('X',float('nan'),100,T+pd.Timedelta(days=2),'missing')
