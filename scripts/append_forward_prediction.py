import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from v4_train import dataset,OUT,predict_neural
from quant_platform.models.sidgtn import SIDGTN
from quant_platform.features.registry import PRIMARY
from quant_platform.research.observation import append_forward
from quant_platform.research.governance import verify_freeze,digest_file,now,write_json,Ledger
from quant_platform.data.forward import read_forward_bars,latest_inputs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='experiments/v4/bars.parquet');parser.add_argument('--dry-check',action='store_true');args=parser.parse_args()
    verify_freeze(json.loads((OUT/'frozen.json').read_text()));timestamp=now();path=OUT/'forward/predictions.jsonl'
    track='diagnostic-failed-sidgtn-'+digest_file(OUT/'frozen_model/sidgtn_seed11/model.pt')[:16]
    if not args.dry_check and any(r['record']['track_id']==track and r['record']['predicted_at'][:10]==timestamp[:10] for r in Ledger(path).rows()):
        Ledger(path).verify();print('DAILY_FORWARD_ALREADY_RECORDED');return
    snapshot=ROOT/args.snapshot;b=read_forward_bars(snapshot,timestamp,require_manifest=snapshot.resolve()!=(OUT/'bars.parquet').resolve())
    latest,p,aid=latest_inputs(b,timestamp);s=joblib.load(OUT/'frozen_model/normalizers.joblib')
    x=s['factor'].transform(latest[PRIMARY].to_numpy(dtype=np.float32)).astype(np.float32)
    price=s['price'].transform(p.reshape(-1,3)).reshape(-1,28,3).astype(np.float32)
    saved=torch.load(OUT/'frozen_model/sidgtn_seed11/model.pt',map_location='cpu',weights_only=True)
    model=SIDGTN(len(PRIMARY),saved['config']['hidden'],saved['config']['dropout']);model.load_state_dict(saved['state_dict']);model.eval()
    o=predict_neural(model,price,x,aid);predictions=[]
    for i,(_,row) in enumerate(latest.iterrows()):
        predictions.append(dict(pair=row.pair,decision_at=str(row.decision_at),anchor_close=float(row.close),quantiles=o['quantiles'][i].tolist(),horizons=[14,28]))
    if args.dry_check:
        write_json(OUT/'forward/input_check_metrics.json',dict(schema_version=1,status='FORWARD_NEW_SNAPSHOT_INFERENCE_CHECKED',snapshot=str(snapshot),data_hash=digest_file(snapshot),pairs=len(predictions),ledger_modified=False))
        print('FORWARD_SNAPSHOT_INFERENCE_CHECKED_NO_LEDGER_CHANGE');return
    append_forward(path,dict(track_id=track,model_hash=digest_file(OUT/'frozen_model/sidgtn_seed11/model.pt'),data_hash=digest_file(snapshot),
             data_snapshot_path=str(snapshot),data_manifest_sha256=digest_file(snapshot.parent/'manifest.json') if (snapshot.parent/'manifest.json').exists() else digest_file(OUT/'frozen.json'),
             venue='kraken',quote_currency='USD',processing_version='v4.1',observed_at=str(latest.decision_at.max()),available_at=str(latest.decision_at.max()),
             predicted_at=timestamp,signal_generated_at=timestamp,earliest_executable_at=str(pd.Timestamp(timestamp).floor('D')+pd.Timedelta(days=1)),
             decision_at=str(latest.decision_at.max()),predictions=predictions,purpose='DIAGNOSTIC_FAILED_MODEL_NO_ALPHA_PROMOTION',order_enabled=False))
    write_json(OUT/'forward/metrics.json',dict(schema_version=1,records=Ledger(path).verify(),track_id=track,financial_performance=False))
    print('FILE_ONLY_FORWARD_PREDICTION_APPENDED')
if __name__=='__main__':main()
