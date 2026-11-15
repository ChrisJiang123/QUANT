"""Read-only V4 weight/label/loss audit; V5 forecasts are scored separately."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,read,CFG
from quant_platform.research.governance import digest_file,write_json,offline_guard
from quant_platform.models.sidgtn import SIDGTN

def calibration(g):
    prob=g.direction_probability.to_numpy();truth=(g.actual.to_numpy()>0).astype(float)
    bins=[]
    for low in np.arange(0,1,.1):
        mask=(prob>=low)&(prob<low+.1 if low<.9 else prob<=1)
        if mask.any():bins.append(dict(low=float(low),n=int(mask.sum()),forecast=float(prob[mask].mean()),actual=float(truth[mask].mean())))
    return dict(brier=float(np.mean((prob-truth)**2)),bins=bins,ece=sum(b['n']*abs(b['forecast']-b['actual']) for b in bins)/len(g))

@offline_guard()
def main():
    torch.set_num_threads(4)
    from v4_train import dataset
    meta,price,future_vol=dataset();normal=read(ROOT/'experiments/v4/training/dev_fold7/normalizer.json')
    old=pd.read_parquet(ROOT/'experiments/v4/dev_predictions.parquet');old=old[old.model.str.startswith('sidgtn_seed')&old.metric_eligible].copy()
    diagnostics=[]
    for (model,h),g in old.groupby(['model','horizon']):
        actual=g.actual.to_numpy();error=np.abs(actual-g.prediction.to_numpy());vol=g.forecast_vol.to_numpy()
        # V4's own frozen target contract is retained for the diagnostic.
        targetvol=future_vol[g.sample_id.to_numpy(),[14,28].index(h)]
        diagnostics.append(dict(model=model,horizon=int(h),samples=len(g),zero_mae=float(np.abs(actual).mean()),mae=float(error.mean()),
            coverage80=float(((g.actual>=g.q10)&(g.actual<=g.q90)).mean()),mean_predicted_annual_vol=float(vol.mean()),mean_realized_annual_vol=float(np.nanmean(targetvol)),
            median_predicted_to_realized_vol_ratio=float(np.nanmedian(vol/np.maximum(targetvol,.01))),
            direction_mean=float((2*g.direction_probability-1).mean()),median_mean=float(g.prediction.mean()),calibration=calibration(g)))
    # Reproduce actual frozen checkpoint output and executed loss, not prose.
    g=old[(old.fold=='dev_fold7')&(old.model=='sidgtn_seed11')&(old.horizon==14)].head(64)
    ids=g.sample_id.to_numpy();m=meta.iloc[ids];savedpath=ROOT/'experiments/v4/training/dev_fold7/sidgtn_seed11/model.pt'
    saved=torch.load(savedpath,map_location='cpu',weights_only=False);cfg=saved['config'];model=SIDGTN(8,cfg['hidden'],cfg['dropout']);model.load_state_dict(saved['state_dict']);model.eval()
    from quant_platform.features.registry import PRIMARY
    x=(m[PRIMARY].to_numpy(dtype=np.float32)-np.array(normal['factor_mean']))/np.array(normal['factor_scale'])
    p=(price[ids]-np.array(normal['price_mean']))/np.array(normal['price_scale'])
    with torch.no_grad():o=model(torch.tensor(p,dtype=torch.float32),torch.tensor(x,dtype=torch.float32),torch.tensor(m.asset_id.to_numpy()))
    y=torch.tensor(m[['y14','y28']].to_numpy(dtype=np.float32));v=torch.tensor(future_vol[ids]);err=y[:,:,None]-o['quantiles'];q=torch.tensor([.1,.5,.9])
    terms=dict(pinball=float(torch.maximum(q*err,(q-1)*err).mean()),direction_BCE=float(F.binary_cross_entropy_with_logits(o['direction_logit'],(y>0).float())),vol_Huber=float(F.smooth_l1_loss(o['volatility'],v)),tail_Huber_to_return=float(F.smooth_l1_loss(o['tail'],y)))
    reproduce_error=float(np.max(np.abs(o['quantiles'][:,0,1].numpy()-g.prediction.to_numpy())))
    if reproduce_error>2e-5:raise ValueError('old frozen weights do not reproduce predictions')
    timing=dict(candle_close_as_signal_endpoint=True,old_decision_at_equals_close_boundary=bool((old.decision_at==old.candle_ts+pd.Timedelta(days=1)).all()),
        v5_rule='observation close+6h; execute observation+2d open; future open labels',old_label_warning='Old close-to-close label includes return before the newly delayed executable open. Old diagnostics are exploratory; four-grid retimes forecasts to V5 execution without relabelling as clean final.')
    histories=[]
    for path in (ROOT/'experiments/v4/training').glob('dev_fold*/sidgtn_seed*/training.jsonl'):
        h=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        histories.append(dict(fold=path.parent.parent.name,seed=path.parent.name,epochs=len(h),last_train=h[-1]['train_loss'],best_inner=min(r['inner_stop_loss'] for r in h),last_inner=h[-1]['inner_stop_loss']))
    result=dict(old_predictions_sha256=digest_file(ROOT/'experiments/v4/dev_predictions.parquet'),old_weight_sha256=digest_file(savedpath),weight_reproduction_max_error=reproduce_error,
        reproduced_loss_terms=terms,actual_loss='pinball + .1 BCE + .1 annual-vol Huber + .1 tail Huber to realized RETURN, not .05 pinball',
        configured_tail_correction='V5 uses registered normalized q05 pinball; V4 source/weights unchanged',timing=timing,
        normalizer=normal,metrics=diagnostics,learning_curves=histories,
        conclusion_scope='Associations and dimensional audits, not causal attribution of failure. Separate direction/median/vol financial quadrants required.',
        effective_overlap='Horizon labels overlap daily; HAC h-1 and date-level aggregation required. V5 weekly anchored training; no claim that overlapping rows are independent.')
    write_json(OUT/'legacy_diagnostic.json',result);print(json.dumps(dict(reproduced_max_error=reproduce_error,loss_terms=terms,metric_groups=len(diagnostics),histories=len(histories))))

if __name__=='__main__':main()
