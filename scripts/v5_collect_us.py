"""One bounded append-only U.S. public capture; no accounts, orders or bot."""
import json,sys,uuid,urllib.parse
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,read
from quant_platform.research.governance import now,write_json,digest_file,Ledger
from v5_public_audit import get

def closed_prices(response,spec,received_at,source):
    received=pd.Timestamp(received_at);rows=[]
    values=response if isinstance(response,list) else response.get('data',[])
    for item in values:
        day=pd.Timestamp(item['date'],tz='UTC');observed=day+pd.Timedelta(days=1);o=item.get('ohlc',{})
        if observed>received or any(o.get(k) is None for k in ['open','high','low','close']):continue
        rows.append(dict(pair=spec['pair'],instrument_id=spec['instrument_id'],product_id=spec['product_id'],event_at=day,observed_at=observed,available_at=received,received_at=received,
            source=source,processing_version='v5-us-append-capture-1',grade='REAL_US_PERP_PARTIAL',**{k:float(o[k]) for k in ['open','high','low','close']}))
    return rows

def main():
    at=pd.Timestamp(now());folder=OUT/'us_collections'/(at.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True,exist_ok=False)
    coverage=[];receipts=[]
    for spec in read(OUT/'instrument_specs.json')['instruments']:
        asset=spec['pair'];pid=spec['product_id'];begin=spec['first_trading_day']+'T00:00:00Z'
        price_url=f'https://bitnomial.com/exchange/api/web/charts/price/{pid}?'+urllib.parse.urlencode(dict(begin_time=begin,end_time='now'))
        funding_url='https://bitnomial.com/exchange/api/v1/funding-rates/?'+urllib.parse.urlencode(dict(product_id=pid,begin_time=(at-pd.Timedelta(days=7)).isoformat(),end_time=at.isoformat(),limit=1000,order='asc'))
        price=get(price_url,folder/(asset+'_prices.raw'));funding=get(funding_url,folder/(asset+'_funding.raw'))
        write_json(folder/(asset+'_prices.receipt.json'),price,immutable=True);write_json(folder/(asset+'_funding.receipt.json'),funding,immutable=True);receipts.extend([price,funding])
        p=closed_prices(read(folder/(asset+'_prices.raw')),spec,price['received_at'],price_url) if price['status']=='ACQUIRED' else []
        if p:pd.DataFrame(p).to_parquet(folder/(asset+'_closed_prices.parquet'),index=False)
        rates=[]
        if funding['status']=='ACQUIRED':
            response=read(folder/(asset+'_funding.raw'));values=response.get('data',[]) if isinstance(response,dict) else response
            for r in values:
                event=pd.Timestamp(r['interval_end']);received=pd.Timestamp(funding['received_at'])
                if event>received:continue
                rates.append(dict(**r,pair=asset,instrument_id=spec['instrument_id'],event_at=event,observed_at=event,available_at=received,received_at=received,
                    source=funding_url,processing_version='v5-us-append-capture-1',grade='REAL_US_PERP_OBSERVED_BACKFILL'))
        if rates:pd.DataFrame(rates).to_parquet(folder/(asset+'_funding.parquet'),index=False)
        coverage.append(dict(pair=asset,prices=len(p),funding=len(rates),price_start=str(min(x['event_at'] for x in p)) if p else None,price_end=str(max(x['event_at'] for x in p)) if p else None,
            historical_margin_verified=False,full_mark_intraday_verified=False,evidence='REAL_US_PERP_PARTIAL',current_price_fresh=bool(p and at-max(x['observed_at'] for x in p)<=pd.Timedelta(days=2))))
    result=dict(captured_at=at.isoformat(),coverage=coverage,receipts=receipts,history_is_backfill=True,financial_evidence=False,us_status='BLOCKED_US_FORWARD',
        reason='Current public capture cannot replace missing historical dollar margin, continuous mark and sufficient closed bars',files={p.name:digest_file(p) for p in folder.iterdir() if p.is_file()})
    write_json(folder/'result.json',result,immutable=True);write_json(OUT/'us_collection_latest.json',dict(result=str((folder/'result.json').relative_to(ROOT)),hash=digest_file(folder/'result.json')))
    Ledger(OUT/'collection_ledger.jsonl').append(dict(kind='PUBLIC_US_CAPTURE',at=at.isoformat(),result=str((folder/'result.json').relative_to(ROOT)),hash=digest_file(folder/'result.json')))
    print(json.dumps(dict(result=str(folder/'result.json'),coverage=coverage,us_status=result['us_status'])))

if __name__=='__main__':main()
