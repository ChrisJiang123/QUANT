# SPDX-License-Identifier: GPL-3.0-or-later
"""Receipt-time adapter for the same offline Freqtrade engine and weight callbacks."""
import importlib.util
from pathlib import Path
import pandas as pd

_path = Path(__file__).resolve().parents[1]/'strategies/V4TargetWeights.py'
_spec = importlib.util.spec_from_file_location('frozen_weights', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

class V4TargetWeights(_module.V4TargetWeights):
    def bot_start(self, **kwargs):
        self.signals = {}
        self.opens = {}
        self.volumes = {}
        for pair in self.config['exchange']['pair_whitelist']:
            s = pd.read_parquet(Path(self.config['v4_signals'])/(pair.replace('/','_')+'.parquet'))
            if not ((s.available_at <= s.signal_generated_at) &
                    (s.signal_generated_at <= s.earliest_executable_at) &
                    (s.observed_candle_ts+pd.Timedelta(days=1) <= s.available_at)).all():
                raise ValueError('receipt-time signal violation')
            if s.candle_ts.duplicated().any(): raise ValueError('duplicate signal clock')
            self.signals[pair] = s
            b = pd.read_feather(Path(self.config['v4_data_dir'])/(pair.replace('/','_')+'-1d.feather')).set_index('date')
            self.opens[pair] = b.open
            self.volumes[pair] = b.volume
        self.audit_path = Path(self.config['v4_audit_path'])
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _signal(self, pair, current_time):
        s = super()._signal(pair, current_time)
        # A missed daily observation never authorizes a fresh entry or adjustment.
        if s is None or s.earliest_executable_at != current_time: return None
        return s

    def _bar_present(self, pair, current_time):
        return (current_time in self.opens[pair].index and
                pd.notna(self.opens[pair].loc[current_time]) and
                self.volumes[pair].loc[current_time] > 0)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        return self._bar_present(pair, current_time) and self._signal(pair, current_time) is not None

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force,
                           exit_reason, current_time, **kwargs):
        return self._bar_present(pair, current_time)
