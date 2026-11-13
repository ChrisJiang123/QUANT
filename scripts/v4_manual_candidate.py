import json,sys
from pathlib import Path
import pandas as pd
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.features.research_imports import import_candidate_file,risk_review_template
from quant_platform.research.governance import write_json,Ledger,digest_file,offline_guard

def main():
    out=ROOT/'experiments/v4';folder=out/'auto_mining';folder.mkdir(exist_ok=True)
    with offline_guard():
        bars=pd.read_parquet(out/'bars.parquet');frame=bars[(bars.pair=='XBTUSD')&(bars.available_at<pd.Timestamp('2025-01-01',tz='UTC'))]
        path=folder/'manual_candidates.json';write_json(path,dict(schema_version=1,expressions=['mean(close,28)/mean(close,120)-1']),immutable=True)
        result=import_candidate_file(path,frame,folder/'candidates.jsonl')
        write_json(folder/'metrics.json',dict(schema_version=1,manual_mode=True,auto_mining_enabled=False,results=result,
             data_snapshot_sha256=digest_file(out/'bars.parquet'),rows=len(frame),status='PENDING_HUMAN_APPROVAL',financial_evaluation=False,
             candidate_chain_records=Ledger(folder/'candidates.jsonl').verify()))
        write_json(out/'static_review/metrics.json',risk_review_template('PARTIAL_PIT_COVERAGE','NO_GO_ALPHA'))
    print('MANUAL_OFFLINE_CANDIDATE_RECORDED_NO_PROMOTION')
if __name__=='__main__':main()
