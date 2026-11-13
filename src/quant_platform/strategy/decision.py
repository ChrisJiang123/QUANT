import numpy as np

def target_weights(score,vol,crowding=1.,event=1.,target_vol=.15,w_max=.5,source='price'):
    if source not in {'price','rule','ridge','elasticnet','lightgbm','sidgtn'}: raise ValueError('unapproved signal source')
    s=np.asarray(score,dtype=float);v=np.asarray(vol,dtype=float)
    if not np.isfinite(s).all() or not np.isfinite(v).all() or (v<=0).any(): return np.zeros_like(s)
    w=np.minimum(w_max,np.clip(s,0,1)*np.clip(crowding,0,1)*np.clip(event,0,1)*target_vol/v)
    return w/max(1.,float(w.sum()))

def historical_risk(weights,quote_volume,notional=10000.,participation=.001):
    w=np.asarray(weights);vol=np.asarray(quote_volume)
    reasons=[]
    if (w<0).any() or (w>.5+1e-9).any() or w.sum()>1+1e-9: reasons.append('EXPOSURE_REJECT')
    if (~np.isfinite(vol)).any() or (w*notional>vol*participation).any(): reasons.append('LIQUIDITY_PROXY_REJECT')
    return dict(status='REJECT' if reasons else 'PASS_RESEARCH_ONLY',reasons=reasons,live_authorized=False)
