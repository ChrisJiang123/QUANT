import argparse,json,sys
import pandas as pd
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import Ledger,write_json,now,digest_file
from quant_platform.data.forward import read_forward_bars,matured_labels

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='experiments/v4/bars.parquet');args=parser.parse_args()
    out=ROOT/'experiments/v4';source=Ledger(out/'forward/predictions.jsonl');source.verify();settled=Ledger(out/'forward/settlements.jsonl');settled.verify()
    snapshot=ROOT/args.snapshot;is_original=snapshot.resolve()==(out/'bars.parquet').resolve()
    if is_original:
        frozen=json.loads((out/'frozen.json').read_text())
        if digest_file(snapshot)!=frozen['files'][str(snapshot.resolve())]:raise ValueError('frozen price snapshot changed')
    bars=read_forward_bars(snapshot,now(),require_manifest=not is_original);count=0
    existing={r['record']['prediction_hash'] for r in settled.rows()}
    for row in source.rows():
        if row['hash'] in existing:continue
        labels=matured_labels(row['record']['predictions'],bars,now())
        if len(labels)==2*len(row['record']['predictions']):settled.append(dict(prediction_hash=row['hash'],labels=labels));count+=1
    write_json(out/'forward/settlement_metrics.json',dict(schema_version=1,new_settlements=count,settlements=settled.verify(),pending_predictions=len(source.rows())-len(settled.rows())))
    print('FORWARD_SETTLEMENT_CHECKED',count)
if __name__=='__main__':main()
