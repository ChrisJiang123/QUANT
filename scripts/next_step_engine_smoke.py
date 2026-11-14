"""Isolated historical fixtures. These outputs are never forward/Alpha evidence."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from forward_finance import simulate
from quant_platform.research.governance import write_json
from quant_platform.research.rule_forward import DAY

def main():
    folder=ROOT/'experiments/next_step_20261113/fixture_engine_smoke_v2'
    times=pd.date_range('2020-01-01',periods=200,tz='UTC')
    frames=[]
    for j,pair in enumerate(['ETHUSD','XBTUSD']):
        c=(100+np.arange(200)*.2)*(j+1)
        frames.append(pd.DataFrame(dict(pair=pair,candle_ts=times,open=c-.1,
                       high=c+1,low=c-1,close=c,volume=100000.,available_at=times+DAY)))
    bars=pd.concat(frames,ignore_index=True); observations=[]
    for i,start in enumerate(times[-5:]):
        generated=start-DAY+pd.Timedelta(hours=2)
        assets=[dict(pair=p,candle_ts=str(start-2*DAY),target_weight=.2) for p in ['ETHUSD','XBTUSD']]
        observations.append(dict(hash=str(i),record=dict(eligible_clean=True,
            earliest_executable_at=str(start),generated_at=str(generated),received_at=str(generated),
            rebalance=i==0,assets=assets)))
    result=simulate(bars,observations,folder)
    assert result['status']=='OFFLINE_ENGINE_RECONSTRUCTION'
    assert len(result['performance'])==9
    for k,m in result['performance'].items():
        assert m['effective_cost_per_side'] in [.0035,.0085,.012]
        if k.startswith('rule/'):
            assert m['trades']==2 and m['filled_orders']==4
    write_json(folder/'fixture_only_metrics.json',dict(fixture_only=True,financial_evidence=False,
               all_nine_guarded_engine_runs_pass=True,result=result))
    print('FIXTURE_ONLY_NINE_ENGINE_RUNS_PASS')

if __name__=='__main__': main()
