"""Independent same-seed retraining check; no new hyperparameters or terminal evaluation."""
import json
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from v4_bootstrap import ROOT
from v4_train import dataset,neural_fit,predict_neural,CONFIG,OUT
from quant_platform.features.registry import PRIMARY
from quant_platform.models.sidgtn import SIDGTN
from quant_platform.research.governance import write_json,digest_file,Ledger,offline_guard

def scaler(mean,scale):
    s=StandardScaler();s.mean_=np.array(mean);s.scale_=np.array(scale);s.var_=s.scale_**2;s.n_features_in_=len(mean);return s

def main():
    folder=OUT/'reproduction/dev_fold0_seed11';folder.mkdir(parents=True,exist_ok=True)
    with offline_guard():
        m,p,v=dataset();n=json.loads((OUT/'training/dev_fold0/normalizer.json').read_text());cutoff=pd.Timestamp(n['train_cutoff']);inner_start=cutoff-pd.Timedelta(days=180)
        train=(m.label_available28<cutoff)&m.y28.notna()&np.isfinite(v).all(axis=1)
        inner=train&(m.label_available28<inner_start-pd.Timedelta(days=28));stop=train&(m.decision_at>=inner_start)
        x=scaler(n['factor_mean'],n['factor_scale']).transform(m[PRIMARY].to_numpy(dtype=np.float32)).astype(np.float32)
        price=scaler(n['price_mean'],n['price_scale']).transform(p.reshape(-1,3)).reshape(p.shape).astype(np.float32)
        y=m[['y14','y28']].to_numpy(dtype=np.float32);aid=m.asset_id.to_numpy()
        model,stat=neural_fit(x,price,aid,y,v,inner.to_numpy(),stop.to_numpy(),CONFIG['sidgtn'],folder,11)
        old_path=OUT/'training/dev_fold0/sidgtn_seed11/model.pt';saved=torch.load(old_path,map_location='cpu',weights_only=True)
        original=SIDGTN(len(PRIMARY),saved['config']['hidden'],saved['config']['dropout']);original.load_state_dict(saved['state_dict']);original.eval()
        current=model.cpu().state_dict();diff=max(float((current[k]-saved['state_dict'][k]).abs().max()) for k in current)
        idx=np.flatnonzero(stop)[-64:];a=predict_neural(model.cpu(),price[idx],x[idx],aid[idx]);b=predict_neural(original,price[idx],x[idx],aid[idx])
        maximum=float(np.max(np.abs(a['quantiles']-b['quantiles'])));passed=bool(np.allclose(a['quantiles'],b['quantiles'],atol=1e-5,rtol=1e-4))
        result=dict(schema_version=1,seed=11,reference_sha256=digest_file(old_path),weight_max_absolute_difference=diff,
                    prediction_max_absolute_difference=maximum,prediction_atol=1e-5,prediction_rtol=1e-4,passed=passed,terminal_evaluated=False,training=stat)
        write_json(OUT/'reproduction/metrics.json',result);Ledger(OUT/'ledger.jsonl').append(dict(kind='REPRODUCTION_CHECK',**result))
    if not passed:raise SystemExit('same-seed reproduction tolerance failed')
    print('INDEPENDENT_SAME_SEED_RETRAINING_REPRODUCED')
if __name__=='__main__':main()
