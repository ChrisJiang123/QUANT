"""Final manifests, source snapshot, observation planning and honest source rights metadata."""
import json,sys,zipfile
from pathlib import Path
import pyarrow.parquet as pq
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import write_json,digest_file,verify_freeze,Ledger

def main():
    out=ROOT/'experiments/v4';frozen=json.loads((out/'frozen.json').read_text());verify_freeze(frozen);Ledger(out/'ledger.jsonl').verify()
    codezip=out/'frozen_code.zip'
    if not codezip.exists():
        with zipfile.ZipFile(codezip,'x',zipfile.ZIP_DEFLATED) as z:
            for name in frozen['files']:
                p=Path(name)
                if p.suffix in {'.py','.json','.md'} and not p.is_relative_to(out/'frozen_model'):z.write(p,p.relative_to(ROOT).as_posix())
    write_json(out/'frozen_code_receipt.json',dict(schema_version=1,archive=str(codezip.relative_to(ROOT)),sha256=digest_file(codezip),frozen_manifest_sha256=digest_file(out/'frozen.json')))
    score=json.loads((out/'scorecard/metrics.json').read_text());cfg=json.loads((ROOT/'configs/v4.json').read_text())
    windows=score['minimum_independent_windows'];h=cfg['purge_days'];anchor_days=windows*h;total=anchor_days+h
    write_json(out/'planning/metrics.json',dict(schema_version=1,existing_fresh_windows=score['fresh_independent_windows'],additional_split_days=score['required_additional_days'],
          independent_window_days=h,required_windows=windows,new_forward_anchor_days=anchor_days,maturation_days=h,new_forward_total_days=total,
          approx_calendar_months=total/30.4375,qualification='sampling-duration estimate only; failed DEV gate still requires a new model/hypothesis and preregistration'))
    contracts=[];folder=out/'contracts';folder.mkdir(exist_ok=True)
    for p in sorted(out.rglob('*.parquet')):
        meta=pq.read_metadata(p);sha=digest_file(p);record=dict(schema_version=1,path=p.relative_to(ROOT).as_posix(),sha256=sha,rows=meta.num_rows,
                   columns=pq.read_schema(p).names,processing_version='v4-research-artifact',financial_performance_source='Freqtrade engine exports only' if 'engine_' in p.name else 'research data/prediction, not execution')
        target=folder/(sha+'.json')
        # Same bytes may appear at multiple paths; key the catalog by path+hash.
        target=folder/(__import__('hashlib').sha256(record['path'].encode()).hexdigest()+'-'+sha+'.json')
        write_json(target,record,immutable=True);contracts.append(record)
    write_json(folder/'metrics.json',dict(schema_version=1,artifacts=contracts,count=len(contracts)))
    rights=[dict(source='Kraken OHLCVT/funding',terms='Official public download; retain publisher URLs and raw archive notices. No relicensing or redistribution right is inferred.',attribution='Kraken',
             url='https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data'),
            dict(source='Deribit DVOL',terms='Public market-data endpoint; publisher terms retained, personal research only.',attribution='Deribit',url='https://docs.deribit.com/'),
            dict(source='Fear & Greed',terms='Publisher attribution required where displayed; no historical first-seen proof claimed.',attribution='Source: Alternative.me',url='https://alternative.me/crypto/fear-and-greed-index/'),
            dict(source='Wikipedia pageviews',terms='Aggregated public metrics; retain Wikimedia API terms/attribution; no article text copied.',attribution='Wikimedia Analytics Pageviews API',url='https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews')]
    write_json(out/'source_rights/metrics.json',dict(schema_version=1,sources=rights,redistribution_review='NOT_AUTHORIZED_OR_ASSUMED'))
    print('FROZEN_DEPENDENCIES_AND_DELIVERY_CONTRACTS_VERIFIED')
if __name__=='__main__':main()
