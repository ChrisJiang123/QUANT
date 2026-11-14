"""Synthetic availability smoke only. Calls FreqAI model.fit, never its bot loop."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/v4/freqai_smoke';OUT.mkdir(exist_ok=True)
def deny(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:raise PermissionError('offline FreqAI smoke')
sys.addaudithook(deny)

def main():
    from freqtrade.freqai.prediction_models.LightGBMRegressor import LightGBMRegressor
    from freqtrade.enums import RunMode
    cfg=dict(user_data_dir=OUT,exchange={'pair_whitelist':['BTC/USD']},timeframe='1d',runmode=RunMode.BACKTEST,
             freqai=dict(enabled=True,identifier='SYNTHETIC_SMOKE_ONLY',feature_parameters={},data_split_parameters={'test_size':0},
                         activate_tensorboard=False,model_training_parameters={'n_estimators':5,'num_leaves':3,'verbosity':-1,'n_jobs':1}))
    model=LightGBMRegressor(cfg);rng=np.random.default_rng(11);x=pd.DataFrame(rng.normal(size=(100,3)),columns=['a','b','c']);y=pd.DataFrame({'target':x.a*.1})
    fitted=model.fit(dict(train_features=x,train_labels=y,train_weights=np.ones(100)),SimpleNamespace(pair='BTC/USD',data_path=OUT))
    result=fitted.predict(x.iloc[:5])
    if not np.isfinite(result).all() or model._threads:raise ValueError('smoke output or no-thread contract failed')
    (OUT/'metrics.json').write_text(json.dumps(dict(schema_version=1,status='SYNTHETIC_FIT_SMOKE_PASSED',
                module='freqtrade.freqai.prediction_models.LightGBMRegressor.LightGBMRegressor.fit',rows=100,predictions=result.tolist(),
                synthetic_test_only=True,financial_performance=False,online_bot_started=False),indent=2))
    print('FREQAI_SYNTHETIC_FIT_SMOKE_PASSED')
if __name__=='__main__':main()
