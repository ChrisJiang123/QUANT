# SPDX-License-Identifier: GPL-3.0-or-later
# External CLI plugin boundary; no Freqtrade code copied into the core package.
import json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

class V4TargetWeights(IStrategy):
    INTERFACE_VERSION=3
    timeframe='1d';can_short=False;startup_candle_count=0
    minimal_roi={};stoploss=-.99;trailing_stop=False;process_only_new_candles=True
    use_exit_signal=True;exit_profit_only=False;position_adjustment_enable=True
    max_entry_position_adjustment=-1
    order_types={'entry':'market','exit':'market','stoploss':'market','stoploss_on_exchange':False}
    order_time_in_force={'entry':'GTC','exit':'GTC'}

    def bot_start(self,**kwargs):
        path=Path(self.config['v4_signals'])
        if not path.is_dir():raise ValueError('missing frozen signals')
        self.signals={pair:pd.read_parquet(path/(pair.replace('/','_')+'.parquet')) for pair in self.config['exchange']['pair_whitelist']}
        for pair,f in self.signals.items():
            if not ((f.available_at<=f.earliest_executable_at)&(f.available_at>=f.candle_ts+pd.Timedelta(days=1))).all():
                raise ValueError('signal PIT violation')
        self.audit_path=Path(self.config['v4_audit_path']);self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        self.opens={pair:pd.read_feather(Path(self.config['v4_data_dir'])/(pair.replace('/','_')+'-1d.feather')).set_index('date')['open'] for pair in self.signals}

    def _equity(self,current_time):
        value=self.wallets.get_total('USD')
        for trade in Trade.get_trades_proxy(is_open=True):
            prices=self.opens[trade.pair]
            if current_time not in prices.index:raise ValueError('missing current opening price')
            value+=trade.amount*float(prices.loc[current_time])
        return value

    def populate_indicators(self,dataframe,metadata):
        s=self.signals[metadata['pair']].rename(columns={'candle_ts':'date'})
        cols=['date','target_weight','decision','gap_mask']
        df=dataframe.merge(s[cols],on='date',how='left',validate='one_to_one')
        df['target_weight']=df.target_weight.fillna(0.)
        df['decision']=df.decision.fillna(False);df['gap_mask']=df.gap_mask.fillna(True)
        return df

    def populate_entry_trend(self,dataframe,metadata):
        dataframe.loc[(dataframe.target_weight>.001)&dataframe.decision&~dataframe.gap_mask&(dataframe.volume>0),'enter_long']=1
        return dataframe

    def populate_exit_trend(self,dataframe,metadata):
        dataframe.loc[(dataframe.target_weight<=.001)&dataframe.decision&~dataframe.gap_mask,'exit_long']=1
        return dataframe

    def _signal(self,pair,current_time):
        f=self.signals[pair];rows=f[(f.earliest_executable_at<=current_time)&(f.candle_ts<current_time)]
        return None if rows.empty else rows.iloc[-1]

    def _audit(self,pair,current_time,desired,actual,action):
        with self.audit_path.open('a') as f:
            f.write(json.dumps({'pair':pair,'time':str(current_time),'desired_stake':float(desired),
                                'actual_stake':float(actual),'action':action})+'\n')

    def custom_stake_amount(self,pair,current_time,current_rate,proposed_stake,min_stake,max_stake,leverage,entry_tag,side,**kwargs):
        signal=self._signal(pair,current_time)
        if signal is None or signal.gap_mask:return 0.
        desired=float(signal.target_weight)*self._equity(current_time)/(1+self.config['fee'])
        actual=min(desired,max_stake)
        if min_stake and actual<min_stake:actual=0.
        self._audit(pair,current_time,desired,actual,'entry')
        return actual

    def adjust_trade_position(self,trade,current_time,current_rate,current_profit,min_stake,max_stake,current_entry_rate,
                              current_exit_rate,current_entry_profit,current_exit_profit,**kwargs):
        if not self.config.get('v4_rebalance',True):return None
        signal=self._signal(trade.pair,current_time)
        if signal is None or not signal.decision or signal.gap_mask or trade.has_open_orders:return None
        age=(current_time-trade.open_date_utc).days
        if age<14:return None
        equity=self._equity(current_time)
        desired=float(signal.target_weight)*equity
        mark=trade.amount*current_rate
        if abs(desired-mark)<.05*equity:return None
        delta=desired-mark
        if delta>0:stake=min(delta,max_stake)
        else:stake=max(-trade.stake_amount,delta*trade.stake_amount/max(mark,1e-9))
        if min_stake and 0<stake<min_stake:return None
        self._audit(trade.pair,current_time,desired,mark+stake,'adjust')
        return stake

    def custom_exit(self,pair,trade,current_time,current_rate,current_profit,**kwargs):
        # Benchmark hold deliberately exempt; strategies follow preregistered maximum holding.
        if self.config.get('v4_max_holding',True) and (current_time-trade.open_date_utc).days>=28:
            return 'max_28d'
        return None

class V4AnalysisWeights(V4TargetWeights):
    """Same external-signal logic, with an explicit warmup for recursive analysis."""
    startup_candle_count=120
