import json
import numpy as np,pandas as pd
from v4_bootstrap import ROOT

def main():
    out=ROOT/'experiments/v4';p=pd.read_parquet(out/'dev_predictions.parquet');p=p[p.metric_eligible];results=[]
    for (model,h),g in p[p.model.str.startswith('sidgtn')].groupby(['model','horizon']):
        y=(g.actual>0).astype(float).to_numpy();prob=g.direction_probability.to_numpy();ece=0.;bins=[]
        for i in range(10):
            mask=(prob>=i/10)&(prob<(i+1)/10 if i<9 else prob<=1)
            if not mask.any():continue
            confidence=float(prob[mask].mean());frequency=float(y[mask].mean());ece+=mask.mean()*abs(confidence-frequency)
            bins.append(dict(lower=i/10,upper=(i+1)/10,samples=int(mask.sum()),probability=confidence,frequency=frequency))
        results.append(dict(model=model,horizon=int(h),samples=len(g),brier=float(np.mean((prob-y)**2)),ece=float(ece),
               q10_q90_coverage=float(((g.actual>=g.q10)&(g.actual<=g.q90)).mean()),nominal_coverage=.8,
               median_interval_width=float((g.q90-g.q10).median()),bins=bins,fitted_on_validation=False))
    folder=out/'calibration';folder.mkdir(exist_ok=True);(folder/'metrics.json').write_text(json.dumps(dict(schema_version=1,results=results),indent=2))
    print('CALIBRATION_MEASURED_NO_POSTHOC_FIT')
if __name__=='__main__':main()
