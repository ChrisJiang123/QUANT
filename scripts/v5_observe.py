"""One-shot public-data research observation; no broker or trading process."""
import argparse,json,sys,uuid,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,CFG,read,registered,status
from quant_platform.v5.model import V5SIDGTN
from quant_platform.v5.strategy import Controller
from quant_platform.v5.book import Book
from quant_platform.v5.replay import specifications
from quant_platform.research.governance import now,write_json,digest_file,Ledger

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-check',action='store_true');args=parser.parse_args();registered()
    verified=read(OUT/'gpu_verification.json');jobreceipt=read(OUT/'gpu_job.json');base=ROOT/jobreceipt['job_root'];job=read(base/'job.json')
    returned=ROOT/verified['return_root'];manifest=read(returned/'return_manifest.json')
    spec=importlib.util.spec_from_file_location('v5_observation_worker',base/'ops/v5_worker_entry.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    torch.set_num_threads(4);raw=dict(np.load(base/'dataset.npz',allow_pickle=False));meta=pd.read_parquet(OUT/'data/samples.parquet')
    selected=meta.groupby('pair').tail(1).index.to_numpy();batchmeta=meta.loc[selected];fold=next(x for x in job['folds'] if x['name']=='final_dev_fit')
    at=pd.Timestamp(now());rows=[]
    for variant in job['variants']:
        for seed in job['seeds']:
            path=returned/'final_dev_fit'/variant/f'seed{seed}'/'model.pt'
            if digest_file(path)!=manifest['files'][str(path.relative_to(returned))]:raise ValueError('forward weight changed')
            saved=torch.load(path,map_location='cpu',weights_only=False);model=V5SIDGTN(**saved['constructor']);model.load_state_dict(saved['state_dict'])
            # Reuse saved train-only scalers, not a refit on forward features.
            arrays={k:v.copy() for k,v in raw.items()}
            for name,maskname in [('price',None),('factors',None),('aux','aux_mask'),('nodes','node_mask'),('sentiment','sentiment_mask')]:
                scaler=saved['scalers'][name];x=(arrays[name]-np.array(scaler['mean']))/np.array(scaler['scale'])
                if maskname:
                    mask=arrays[maskname];x=np.where(mask if mask.shape==x.shape else mask[...,None],x,0.)
                arrays[name]=x.astype(np.float32)
            pred=worker.infer(model,arrays,selected,'cpu')
            for j,(i,m) in enumerate(batchmeta.iterrows()):
                stale=at-pd.Timestamp(m.candle_ts)>pd.Timedelta(days=2)
                rows.append(dict(pair=m.pair,variant=variant,seed=seed,feature_candle=m.candle_ts,feature_available_at=m.available_at,inferred_at=at.isoformat(),
                    earliest_executable_at=(max(at,pd.Timestamp(m.available_at)).floor('D')+pd.Timedelta(days=1)).isoformat(),
                    state='BLOCKED_STALE_PROXY_FEATURES' if stale else 'PROXY_RESEARCH_ONLY',weight_sha256=digest_file(path),
                    source='ACTUAL_VERIFIED_SIDGTN_CPU_INFERENCE',horizons=CFG['horizons'],direction=pred['direction_score'][j].tolist(),annual_vol=pred['volatility'][j].tolist(),
                    quantiles=pred['quantiles'][j].tolist(),tail=pred['tail'][j].tolist(),modality_mask=pred['modality_mask'][j].tolist(),fusion_weights=pred['fusion_weights'][j].tolist(),
                    realized_label=None,financial_evidence=False))
    # U.S. prices acquired today remain stale/sparse; they are raw collection
    # evidence only. No synthetic U.S. quote, flat curve or fake trade is added.
    us=read(OUT/'us_data/coverage.json')
    if (OUT/'us_collection_latest.json').exists():
        pointer=read(OUT/'us_collection_latest.json');path=ROOT/pointer['result']
        if digest_file(path)!=pointer['hash']:raise ValueError('public collection changed')
        us=read(path)
    us_eligible=all(r['historical_margin_verified'] and r['full_mark_intraday_verified'] and at-pd.Timestamp(r['price_end'],tz='UTC')<=pd.Timedelta(days=2) for r in us['coverage'])
    us_state='US_COLLECTION_ELIGIBLE_NOT_FINANCIALLY_VALIDATED' if us_eligible else 'BLOCKED_US_FORWARD'
    prepared=pd.read_pickle(OUT/'replay_inputs.pkl');rule_rows={a:max(r.items(),key=lambda x:x[0])[1] for a,r in prepared[0].items()}
    proxy_prices={a:r['close'] for a,r in rule_rows.items()};control=Controller(CFG,'carry','BASE');book=Book(specifications(),CFG['initial_equity'],**CFG['costs']['BASE'])
    plan=control.plan(rule_rows,proxy_prices,book,at)
    rule_track=[dict(pair=a,source='ACTUAL_RULE_CONTROLLER',feature_candle=str(r['candle_ts']),feature_available_at=str(r['available_at']),inferred_at=at.isoformat(),
        raw_score=float(r['raw_score']),annual_vol=float(r['annual_vol']),hypothetical_lots_at_stale_proxy_close={s:int(n) for (s,x),n in plan['targets'].items() if x==a},
        execution_eligible=False,state='BLOCKED_STALE_PROXY_FEATURES' if at-pd.Timestamp(r['candle_ts'])>pd.Timedelta(days=2) else 'PROXY_RESEARCH_ONLY',financial_evidence=False) for a,r in rule_rows.items()]
    result=dict(observed_at=at.isoformat(),us_status=us_state,us_coverage=us,model_inference=rows,rule_track=rule_track,
        software_inference_rows=len(rows),qualified_forward_rows=0,clean_28d_windows=0,minimum_paper_milestone_months=3,
        run_mode='DRY_CHECK' if args.dry_check else 'ONE_SHOT_RESEARCH_OBSERVATION',live_enabled=False,order_enabled=False,
        next_required_data='Fresh complete public U.S. BTC/ETH quotes/marks, actual margin/liquidation spec; fresh multimodal proxy feature receipt before eligible forward window.')
    folder=OUT/'observations'/at.strftime('%Y%m%dT%H%M%S%fZ');write_json(folder/'result.json',result,immutable=True)
    if not args.dry_check:
        Ledger(OUT/'forward_ledger.jsonl').append(dict(kind='MODEL_INFERENCE_WITH_DATA_GATE',result=str((folder/'result.json').relative_to(ROOT)),result_hash=digest_file(folder/'result.json'),at=at.isoformat(),us_status=us_state,qualified_rows=0))
        status('V5-OBS','IMPLEMENTED_AND_TESTED',[str((folder/'result.json').relative_to(ROOT))],run='COMPLETED_DATA_GATE_BLOCKED',financial='BLOCKED_US_FORWARD',reason='real model inference verified; sparse/stale U.S. prices and old proxy features cannot start a clean financial window')
    write_json(OUT/'observation_latest.json',dict(result=str((folder/'result.json').relative_to(ROOT)),hash=digest_file(folder/'result.json')))
    print(json.dumps(dict(us_status=us_state,inference_rows=len(rows),qualified_forward_rows=0,result=str(folder/'result.json'))))

if __name__=='__main__':main()
