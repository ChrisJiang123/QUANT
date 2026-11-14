import numpy as np
from scipy.stats import norm,skew,kurtosis

def dm_hac(difference,horizon):
    d=np.asarray(difference,dtype=float); n=len(d)
    if n<=horizon+2:return {'statistic':None,'p_one_sided':None,'n_dates':n}
    e=d-d.mean(); var=np.dot(e,e)/n
    for lag in range(1,horizon):var+=2*(1-lag/horizon)*np.dot(e[lag:],e[:-lag])/n
    se=np.sqrt(max(var,1e-18)/n);z=float(d.mean()/se)
    return {'statistic':z,'p_one_sided':float(norm.sf(z)),'n_dates':n,'hac_lag':horizon-1}

def holm(values):
    order=np.argsort(values);out=np.empty(len(values));last=0.
    for i,j in enumerate(order):last=max(last,min(1.,(len(values)-i)*values[j]));out[j]=last
    return out.tolist()

def sharpe(r):
    r=np.asarray(r);return float(np.sqrt(365)*r.mean()/r.std(ddof=1)) if len(r)>1 and r.std(ddof=1)>0 else 0.

def paired_bootstrap(a,b,block=28,draws=1000,seed=2026):
    a=np.asarray(a);b=np.asarray(b);n=min(len(a),len(b));rng=np.random.default_rng(seed)
    if n<block*2:return {'lower90':None,'upper90':None,'independent_blocks':n//block}
    samples=[]
    for _ in range(draws):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=np.concatenate([(np.arange(block)+s)%n for s in starts])[:n]
        samples.append(sharpe(a[idx])-sharpe(b[idx]))
    return {'lower90':float(np.quantile(samples,.05)),'upper90':float(np.quantile(samples,.95)),'independent_blocks':n//block,'block_length':block}

def deflated_sharpe(r,trials):
    r=np.asarray(r); n=len(r)
    if n<56 or r.std()==0:return None
    # Effective observations conservatively use 28-day independent blocks.
    neff=n//28; sr=r.mean()/r.std(ddof=1); trials=max(2,trials)
    gamma=.5772156649015329
    expected=(1-gamma)*norm.ppf(1-1/trials)+gamma*norm.ppf(1-1/(trials*np.e))
    benchmark=expected/np.sqrt(neff-1)
    denom=np.sqrt(max(1e-9,1-skew(r)*sr+(kurtosis(r,fisher=False)-1)*sr*sr/4))
    return float(norm.cdf((sr-benchmark)*np.sqrt(neff-1)/denom))

def forecast_metrics(predictions):
    import pandas as pd
    result={};ps=[];keys=[]
    for (model,h),g in predictions.groupby(['model','horizon']):
        errors=(g.actual-g.prediction).abs();zero=g.actual.abs();mae=float(errors.mean());z=float(zero.mean())
        daily=g.assign(diff=zero-errors).groupby('decision_at')['diff'].mean()
        item={'mae':mae,'zero_mae':z,'mae_skill':1-mae/z if z else 0.,'samples':len(g),
              'independent_windows':len(daily)//int(h),'dm':dm_hac(daily.to_numpy(),int(h))}
        if {'q10','q90'}.issubset(g.columns) and g.q10.notna().all():
            item['q10_q90_coverage']=float(((g.actual>=g.q10)&(g.actual<=g.q90)).mean())
        result[f'{model}/{h}']=item
        ps.append(item['dm']['p_one_sided'] or 1.);keys.append(f'{model}/{h}')
    for k,p in zip(keys,holm(ps)):
        result[k]['p_holm']=p;result[k]['G1_horizon_pass']=result[k]['mae_skill']>0 and p<.05
    return result
