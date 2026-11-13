"""PIT source adapters and event masks. Source failure returns a masked group."""
import pandas as pd
import numpy as np
from quant_platform.data.pit import pit_join

LAGS={'funding':0,'dvol':0,'fear_greed':1,'wikipedia':3}
class ProspectiveSource:
    def __init__(self,name):
        if name not in LAGS:raise ValueError('source not registered')
        self.name=name
    def normalize(self,frame):
        f=frame.copy();f['observed_at']=pd.to_datetime(f['observed_at'],utc=True)
        f['available_at']=f.observed_at+pd.Timedelta(days=LAGS[self.name])
        f['processing_version']='v4-prospective-1';f['source']=self.name
        return f
    def join(self,decisions,frame=None):
        if frame is None or frame.empty:return pd.DataFrame({'decision_at':decisions,'value':np.nan,'source_mask':True})
        out=pit_join(decisions,self.normalize(frame));out['source_mask']=out.value.isna()|((out.decision_at-out.available_at)>pd.Timedelta(days=3))
        out.loc[out.source_mask,'value']=np.nan;return out

def calendar_features(records,decision_at,window_days=2):
    d=pd.Timestamp(decision_at)
    eligible=[]
    for record in records:
        announced=pd.Timestamp(record['announced_at']);event=pd.Timestamp(record['event_at'])
        if announced.tzinfo is None or event.tzinfo is None:raise ValueError('PIT calendar needs UTC')
        if announced<=d and record.get('scheduled',False) and abs((event-d).total_seconds())<=window_days*86400:eligible.append(record)
    return {'event_window':bool(eligible),'event_coefficient':.5 if eligible else 1.,'count':len(eligible)}
