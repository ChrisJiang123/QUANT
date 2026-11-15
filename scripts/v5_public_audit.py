"""Bounded anonymous official metadata snapshots; no accounts or authenticated routes."""
import json,sys,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,read,status
from quant_platform.research.governance import now,write_json,digest_file

SOURCES={
 'kraken_us_perps':'https://support.kraken.com/articles/us-perpetual-futures',
 'kraken_us_fees':'https://support.kraken.com/articles/us-futures-fees',
 'kraken_us_specs':'https://support.kraken.com/articles/contract-specifications',
 'kraken_us_maintenance':'https://support.kraken.com/articles/maintenance-windows-us-perpetual-futures',
 'bitnomial_products':'https://bitnomial.com/exchange/api/v1/prod/product/data/',
 'bitnomial_specs':'https://bitnomial.com/exchange/api/v1/prod/product/specs/',
 'bitnomial_funding_docs':'https://bitnomial.com/exchange/docs/api/rest/funding-rates/',
 'bitnomial_products_docs':'https://bitnomial.com/exchange/docs/api/rest/products/',
 'bitnomial_charts_docs':'https://bitnomial.com/exchange/docs/api/rest/charts/',
 'freqtrade_leverage':'https://raw.githubusercontent.com/freqtrade/freqtrade/2026.10/docs/leverage.md',
 'freqtrade_backtesting':'https://raw.githubusercontent.com/freqtrade/freqtrade/2026.10/docs/backtesting.md'}

def get(url,path,limit=8*1024*1024):
    record=dict(url=url,request_at=now(),anonymous=True,proxy=False)
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url,headers={'User-Agent':'QUANT offline research metadata V5'}),timeout=30) as r:
            raw=r.read(limit+1); record['received_at']=now(); record['http_status']=r.status
        if len(raw)>limit:raise ValueError('response exceeds bound')
        path.write_bytes(raw); record.update(status='ACQUIRED',sha256=digest_file(path),bytes=len(raw),file=str(path.relative_to(ROOT)))
    except Exception as exc:record.update(status='BLOCKED',error=type(exc).__name__+': '+str(exc))
    return record

def main():
    folder=OUT/'public_sources'; folder.mkdir(parents=True,exist_ok=True)
    for name,url in SOURCES.items():
        receipt=folder/(name+'.json')
        if receipt.exists():continue
        r=get(url,folder/(name+'.raw'));write_json(receipt,r,immutable=True)
        print(name,r['status'],flush=True)
    rows=[read(p) for p in folder.glob('*.json')]
    write_json(folder/'index.json',dict(sources=rows,accessed_at=now()))
    status('V5-SPEC','IMPLEMENTED_NOT_RUN',['experiments/v5/public_sources/index.json'],run='OFFICIAL_METADATA_PROBED',financial='UNVERIFIED_VENUE_SPEC',reason='instrument mapping and absolute historical margin/mark rules pending')

if __name__=='__main__':main()
