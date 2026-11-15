import copy
import numpy as np
import pandas as pd
import pytest
from quant_platform.v5.common import CFG
from quant_platform.v5.strategy import trend_features,carry_net_expected,contract_target,Controller
from quant_platform.v5.book import Book
from quant_platform.v5.engine import Engine
from quant_platform.v5.instruments import InstrumentSpec

T=pd.Timestamp('2024-01-01',tz='UTC')
def specs():return {'X':InstrumentSpec('scenario:X','X',1,.01,fixed_fee=0,initial_margin_fraction=.1)}
def row(**kw):return dict(valid=True,available_at=T-pd.Timedelta(hours=18),raw_score=0.,annual_vol=.1,close=100.,ma200=101.,daily_return=0.,funding_mean7_annual=.5,**kw)

def test_equal_group_normalization_and_real_truncated_rule_recompute():
    times=pd.date_range('2020-01-01',periods=400,tz='UTC');p=np.arange(100.,500.)
    b=pd.DataFrame(dict(candle_ts=times,open=p,high=p+.2,low=p-.2,close=p,volume=10000.))
    full=trend_features(b,CFG);cut=trend_features(b.iloc[:300],CFG)
    pd.testing.assert_frame_equal(full.iloc[:300].reset_index(drop=True),cut)
    assert full.raw_score.iloc[-1]==1
    # A current-day extreme cannot become its own Donchian threshold.
    b.loc[399,'close']=1000.;b.loc[399,'high']=1001.
    assert trend_features(b,CFG).raw_score.iloc[-1]==1

def test_carry_zero_trades_when_eight_percent_does_not_cover_taker_cost():
    assert carry_net_expected(.08,28,85,5,1,100,.0113,1)<0
    assert carry_net_expected(.5,28,85,5,1,100,.0113,1)>0
    assert carry_net_expected(.5,30,85,5,1,100,.0113,1)==float('-inf')
    b=Book(specs(),initial=1000);c=Controller(CFG,mode='carry')
    r=row();r['funding_mean7_annual']=.08
    p=c.plan({'X':r},{'X':100},b,T)
    assert p['targets'][('carry','X')]==0
    assert c.rejections[-1]['reason']=='CARRY_NET_COST_GATE'

def test_short_requires_ma200_and_half_score():
    b=Book(specs(),initial=1000);c=Controller(CFG,mode='asymmetric')
    r=row();r['raw_score']=-1.;r['close']=102
    p=c.plan({'X':r},{'X':100},b,T);assert p['targets'][('trend','X')]==0
    r['close']=100;p=c.plan({'X':r},{'X':100},b,T)
    expected=contract_target(-.5,.1,1000,100,specs()['X'],.6,CFG)
    assert p['targets'][('trend','X')]==expected<0

def test_tactical_actual_event_implementation_default_main_off():
    r=row();r.update(daily_return=-.2,funding_prior_high=.001,funding_current=.0003,funding_prior_rank=.95)
    b=Book(specs(),initial=7000)
    main=Controller(CFG,mode='full').plan({'X':r},{'X':100},b,T)
    isolated=Controller(CFG,mode='filtered',tactical_only=True).plan({'X':r},{'X':100},b,T)
    assert main['targets'][('tactical','X')]==0
    assert isolated['targets'][('tactical','X')]>0

def test_carry_two_legs_match_and_rebalance_cash_is_delayed():
    config=copy.deepcopy(CFG);config.update(hard_drawdown=3.,soft_drawdown=2.)
    b=Book(specs(),initial=1000,perp_bps=5,spot_bps=85);e=Engine(b,config);c=Controller(config,mode='carry')
    p=c.plan({'X':row()},{'X':100},b,T)
    e.step(T,{'X':dict(open=100,low=100,high=100,close=100)},**p)
    assert b.spot['X']==abs(b.virtual['carry|X'].quantity)>0
    r=row();r['available_at']=T+pd.Timedelta(hours=6)
    plan=c.plan({'X':r},{'X':111},b,T+pd.Timedelta(days=1))
    e.step(T+pd.Timedelta(days=1),{'X':dict(open=111,low=111,high=111,close=111)},**plan)
    assert b.spot['X']==abs(b.virtual['carry|X'].quantity)
    assert b.transfers and b.transfers[0]['amount']>0
    assert all(pd.Timestamp(t['due'])>T+pd.Timedelta(days=1,hours=23) for t in b.transfers)

def test_normal_band_does_not_block_risk_or_flip():
    b=Book(specs(),initial=10000,perp_bps=0,wallet_fraction=1);e=Engine(b,CFG)
    b.sync({('trend','X'):10},{'X':100},T)
    assert e.constrain({('trend','X'):9},{'X':100},T)[('trend','X')]==10
    assert e.constrain({('trend','X'):9},{'X':100},T,force=True)[('trend','X')]==9
    assert e.constrain({('trend','X'):-1},{'X':100},T)[('trend','X')]==-1

def test_covariance_correlation_check_cannot_claim_formula_realized_vol():
    a=InstrumentSpec('a','A',1,.01,fixed_fee=0);b=InstrumentSpec('b','B',1,.01,fixed_fee=0)
    book=Book({'A':a,'B':b},initial=7000);c=Controller(CFG,mode='buy_hold')
    r=row();r['annual_vol']=.1
    p=c.plan({'A':r,'B':r},{'A':100,'B':100},book,T,covariance=np.array([[.01,.01],[.01,.01]]))
    assert c.covariance[-1]['scale']<1
    w=np.array([p['targets'][('trend',x)]*100/7000 for x in ['A','B']])
    assert np.sqrt(w@np.array([[.01,.01],[.01,.01]])@w)<=.25+1e-8
