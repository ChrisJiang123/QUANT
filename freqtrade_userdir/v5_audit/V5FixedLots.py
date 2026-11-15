# SPDX-License-Identifier: GPL-3.0-or-later
import pandas as pd
from freqtrade.strategy import IStrategy

class V5FixedLots(IStrategy):
    INTERFACE_VERSION=3;timeframe='1d';can_short=False;startup_candle_count=0
    minimal_roi={};stoploss=-.99;process_only_new_candles=True;use_exit_signal=True
    order_types={'entry':'market','exit':'market','stoploss':'market','stoploss_on_exchange':False}
    order_time_in_force={'entry':'GTC','exit':'GTC'}
    def populate_indicators(self,dataframe,metadata):return dataframe
    def populate_entry_trend(self,dataframe,metadata):
        dataframe.loc[dataframe.date==pd.Timestamp('2020-01-02',tz='UTC'),'enter_long']=1;return dataframe
    def populate_exit_trend(self,dataframe,metadata):
        dataframe.loc[dataframe.date==pd.Timestamp('2020-01-06',tz='UTC'),'exit_long']=1;return dataframe
    def custom_stake_amount(self,pair,current_time,current_rate,proposed_stake,min_stake,max_stake,leverage,entry_tag,side,**kwargs):
        # Exactly100 BTC-centi or2 ETH-half contracts, both one underlying coin.
        return min(float(current_rate),max_stake)
