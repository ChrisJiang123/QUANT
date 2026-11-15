"""Independent PIT adapters. Missing values are masks, never invented observations."""
import numpy as np
import pandas as pd

class PITTable:
    def __init__(self,frame,name,grade):
        self.frame=frame.copy();self.name=name;self.grade=grade
        for c in ['event_at','observed_at','available_at','received_at']:
            if c not in self.frame:raise ValueError('four-clock record required: '+c)
            self.frame[c]=pd.to_datetime(self.frame[c],utc=True)
        if self.frame[['event_at','observed_at','available_at','received_at']].isna().any().any():raise ValueError('NaT not allowed')
        if (self.frame.available_at<self.frame.observed_at).any():raise ValueError('availability before observation')
    def asof(self,at):return self.frame[self.frame.available_at<=pd.Timestamp(at)].copy()

def join_asof(decision_times,events,columns,max_age_days=None):
    """Only actual available_at can drive a lookup; no event-date-only joins."""
    left=pd.DataFrame({'decision_at':pd.DatetimeIndex(decision_times)})
    if events.empty:return np.zeros((len(left),len(columns)),np.float32),np.zeros((len(left),len(columns)),bool)
    right=events.sort_values('available_at').drop_duplicates('available_at',keep='last')
    z=pd.merge_asof(left.sort_values('decision_at'),right[['available_at',*columns]],left_on='decision_at',right_on='available_at',direction='backward')
    x=z[columns].to_numpy(dtype=float);mask=np.isfinite(x)
    if max_age_days is not None:
        mask&=((z.decision_at-z.available_at)<=pd.Timedelta(days=max_age_days)).to_numpy()[:,None]
    return np.where(mask,x,0.).astype(np.float32),mask

class FundingAdapter(PITTable):
    def __init__(self,frame,name,grade,linear=True,settlement='USD',rate_period='EXPLICIT_OR_INFERRED_CADENCE'):
        super().__init__(frame,name,grade);self.linear=linear;self.settlement=settlement;self.rate_period=rate_period
        if not {'rate','interval_start','interval_end'}.issubset(frame):raise ValueError('funding intervals required')

class PriceAdapter(PITTable):
    def __init__(self,frame,name,grade,kind='spot'):
        if kind not in {'spot','perp_trade','mark','index'}:raise ValueError('explicit price type required')
        if not {'instrument_id','currency','source','processing_version','price'}.issubset(frame):raise ValueError('versioned price identity required')
        if not np.isfinite(frame.price).all() or (frame.price<=0).any():raise ValueError('invalid price')
        super().__init__(frame,name,grade);self.kind=kind

class SpecificationAdapter(PITTable):
    def __init__(self,frame):
        if not {'instrument_id','effective_at','source_sha256','source'}.issubset(frame):raise ValueError('versioned contract identity required')
        super().__init__(frame,'instrument_specification','REAL_US_PERP_PARTIAL')

class EventCalendarAdapter(PITTable):
    def __init__(self,frame):
        if 'announced_at' not in frame:raise ValueError('historical announcement vintage required')
        super().__init__(frame,'event_calendar','PIT_CALENDAR')
        if (pd.to_datetime(frame.announced_at,utc=True)>self.frame.available_at).any():raise ValueError('announcement not yet available')

class GDELTAdapter(PITTable):
    def __init__(self,frame):
        if not {'asset','news_count','tone','archive_revision'}.issubset(frame):raise ValueError('real mapped news archive required')
        super().__init__(frame,'gdelt','HISTORICAL_EXPLORATORY_UNLESS_RECEIPTS_PROVEN')

class MarginHistoryAdapter(PITTable):
    def __init__(self,frame):
        if not {'instrument_id','initial_dollars','maintenance_dollars','effective_at'}.issubset(frame):raise ValueError('absolute dated margin required')
        super().__init__(frame,'margin_history','REAL_US_PERP')

def future_open_targets(frame,horizons):
    """One complete processing day of lag, strictly future executable opening."""
    f=frame.set_index('candle_ts').sort_index()
    f=f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='D'))
    r=np.log(f.open/f.open.shift())
    y=[];v=[];m=[]
    for h in horizons:
        ret=f.open.shift(-(h+2))/f.open.shift(-2)-1
        valid=f.open.notna().rolling(h+1).sum().shift(-(h+2))==h+1
        vol=np.sqrt(sum(r.shift(-(k+2)).pow(2) for k in range(1,h+1))/h*365)
        mask=valid&ret.notna()&vol.notna()
        y.append(ret.where(mask).to_numpy());v.append(vol.where(mask).to_numpy());m.append(mask.to_numpy())
    return np.stack(y,axis=1),np.stack(v,axis=1),np.stack(m,axis=1),f.index
