"""Content-addressed per-value PIT store; missing values remain explicitly masked."""
from pathlib import Path
import io
import pandas as pd
from quant_platform.research.governance import digest_file
from quant_crypto.storage import immutable_write,sha256

class FeatureStore:
    def __init__(self,root):self.root=Path(root)
    def append(self,rows):
        required={'pair','factor','value','observed_at','available_at','processing_version'}
        if not required.issubset(rows.columns):raise ValueError('feature metadata missing')
        if (rows.available_at<rows.observed_at).any():raise ValueError('availability before observation')
        if rows.available_at.dt.tz is None or rows.observed_at.dt.tz is None:raise ValueError('UTC required')
        data=io.BytesIO();rows.to_parquet(data,index=False);raw=data.getvalue()
        path=self.root/(sha256(raw)+'.parquet');immutable_write(path,raw);return path
    def query(self,pair,factor,decision_at):
        frames=[]
        for path in self.root.glob('*.parquet'):
            if digest_file(path)!=path.stem:raise ValueError('feature store corruption')
            f=pd.read_parquet(path);frames.append(f[(f.pair==pair)&(f.factor==factor)&(f.available_at<=decision_at)])
        return pd.concat(frames).sort_values('available_at') if frames else pd.DataFrame()
