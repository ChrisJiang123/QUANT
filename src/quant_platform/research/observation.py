"""File-only forward/shadow ledgers and conservative observation monitoring."""
import numpy as np
from quant_platform.research.governance import Ledger,digest

def append_forward(path,record):
    required={'track_id','model_hash','data_hash','predicted_at','decision_at','predictions'}
    if not required.issubset(record):raise ValueError('forward contract missing')
    if record.get('source')=='llm_shadow':raise ValueError('shadow cannot enter prediction track')
    ledger=Ledger(path);ledger.verify();day=record['predicted_at'][:10]
    if any(r['record']['track_id']==record['track_id'] and r['record']['predicted_at'][:10]==day for r in ledger.rows()):raise ValueError('daily record already exists')
    return ledger.append(record)

def append_shadow(path,record,now):
    if record.get('backend')!='local_ollama' or record.get('temperature')!=0:raise ValueError('local deterministic model required')
    if any(news['published_at']>now for news in record['news']):raise ValueError('future news')
    if record.get('target_weight') is not None:raise ValueError('shadow has no weights')
    if not -2<=record['stance']<=2 or not 0<=record['confidence']<=1:raise ValueError('invalid judgment')
    ledger=Ledger(path);day=now[:10]
    if any(r['record']['timestamp'][:10]==day for r in ledger.rows()):raise ValueError('shadow day already exists')
    return ledger.append({**record,'timestamp':now,'source':'llm_shadow'})

def psi(reference,current):
    if len(reference)<90 or len(current)<90:return None
    edges=np.unique(np.quantile(reference,np.linspace(0,1,11)));edges[0]=-np.inf;edges[-1]=np.inf
    if len(edges)<3:return None
    a=np.histogram(reference,bins=edges)[0].astype(float);b=np.histogram(current,bins=edges)[0].astype(float)
    a=np.maximum(a/a.sum(),1e-6);b=np.maximum(b/b.sum(),1e-6)
    return float(np.sum((b-a)*np.log(b/a)))

def monitor(source_age_days,price_present,chain_valid,psi_value=None,windows=0,coverage_error=None,excess=None,outside_envelope=False):
    status=[]
    if not chain_valid:status.append('STOP_CHAIN')
    if not price_present:status.append('NO_PREDICTION_PRICE_MISSING')
    if source_age_days>3:status.append('MASK_SOURCE')
    if psi_value is not None and psi_value>.25:status.append('DRIFT')
    if windows>=12 and coverage_error is not None and abs(coverage_error)>.15:status.append('MISCALIBRATED')
    if outside_envelope:status.append('DEGRADED')
    if windows>=12 and excess is not None and excess<0:status.append('RETIRE_TO_RULE')
    return {'status':status or ['OBSERVE_ONLY'],'order_enabled':False,'live_enabled':False}
