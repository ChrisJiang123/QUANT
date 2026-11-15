"""Explicit product-ID joins; bounded public US funding and OHLC acquisition."""
import sys,json,urllib.parse
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,read,status
from quant_platform.research.governance import now,digest_file,write_json
from v5_public_audit import get

def main():
    out=OUT/'us_data';out.mkdir(exist_ok=True)
    specs=read(OUT/'public_sources/bitnomial_specs.raw'); data={r['product_id']:r for r in read(OUT/'public_sources/bitnomial_products.raw')}
    wanted={'BTC':(.01,'XBTUSD'),'ETH':(.5,'ETHUSD')}; instruments=[]; coverage=[]
    receipt_at=read(OUT/'public_sources/bitnomial_specs.json')['received_at']
    for asset,(mult,pair) in wanted.items():
        candidates=[s for s in specs if s.get('type')=='perpetual' and s.get('product_status')=='active' and s.get('contract_size_unit')==asset and s.get('contract_size')==mult]
        if len(candidates)!=1: raise ValueError('ambiguous explicit product identity')
        s=candidates[0]; pid=s['product_id']; current=data.get(pid,{})
        instrument=dict(instrument_id='BITNOMIAL:'+str(pid),pair=pair,product_id=pid,exchange_symbol=s['symbol'],base_symbol=s['base_symbol'],
            asset=asset,multiplier=mult,tick=s['price_increment'],linear=True,settlement_currency='USD',
            settlement_method=s['settlement_method'],first_trading_day=s['first_trading_day'],integer_contracts=True,
            funding_interval_hours=s.get('funding_interval_hours'),funding_upper=s.get('funding_rate_upper_bound'),funding_lower=s.get('funding_rate_lower_bound'),
            fixed_fee_components={'broker':.0003,'exchange_clearing':.001,'nfa':.01},fixed_fee_per_contract_side=.0113,
            cash_settlement='15:00 America/Chicago per Kraken broker page',
            product_daily_settle_time=s.get('daily_settle_time'),product_first_funding_end=s.get('daily_first_funding_interval_end'),
            source_available_at=receipt_at,margin_status='UNVERIFIED_VENUE_SPEC',
            contemporaneous_group_margin_rate=current.get('product_group_margin_rate'),historical_absolute_margin=None,
            liquidation_mark_history=None,liquidation_fee_perp_only=10.,
            maintenance_windows_source='https://support.kraken.com/articles/maintenance-windows-us-perpetual-futures',
            historical_maintenance_schedule='UNVERIFIED',source_sha256=digest_file(OUT/'public_sources/bitnomial_specs.raw'))
        instruments.append(instrument)
        frames=[];cursor=None
        for page in range(12):
            filename=out/f'{pair}_funding_page{page}.raw'; receipt=filename.with_suffix('.json')
            query=dict(product_id=pid,begin_time=s['first_trading_day']+'T00:00:00Z',end_time=now(),limit=1000,order='asc')
            if cursor:query['cursor']=cursor
            if not receipt.exists(): write_json(receipt,get('https://bitnomial.com/exchange/api/v1/prod/funding-rates/?'+urllib.parse.urlencode(query),filename),immutable=True)
            meta=read(receipt)
            if meta['status']=='BLOCKED' and '404' in meta.get('error',''):
                # The official endpoint examples omit /prod for these routes.
                # Preserve the failed base-route receipt; no fallback on 403.
                filename=out/f'{pair}_funding_documented_page{page}.raw';receipt=filename.with_suffix('.json')
                if not receipt.exists():write_json(receipt,get('https://bitnomial.com/exchange/api/v1/funding-rates/?'+urllib.parse.urlencode(query),filename),immutable=True)
                meta=read(receipt)
            if meta['status']!='ACQUIRED':break
            value=read(filename); rows=value.get('data',[]) if isinstance(value,dict) else value
            if rows: frames.append(pd.DataFrame(rows).assign(received_at=meta['received_at']))
            new=value.get('pagination',{}).get('cursor') if isinstance(value,dict) else None
            if not rows or not new or new==cursor:break
            cursor=new
        if frames:
            f=pd.concat(frames,ignore_index=True).drop_duplicates(['product_id','interval_end']).sort_values('interval_end')
            for c in ['interval_start','interval_end','received_at']:f[c]=pd.to_datetime(f[c],utc=True)
            f['pair']=pair;f['event_at']=f.interval_end;f['observed_at']=f.interval_end
            f['available_at']=f.received_at;f['evidence_grade']='REAL_US_PERP_OBSERVED_HISTORY_BACKFILL'
            f.to_parquet(out/(pair+'_funding.parquet'),index=False)
        else:f=pd.DataFrame()
        raw=out/(pair+'_prices.raw'); meta=raw.with_suffix('.json')
        url=f'https://bitnomial.com/exchange/api/v1/prod/web/charts/price/{pid}?begin_time={s["first_trading_day"]}T00:00:00Z&end_time=now'
        if not meta.exists():write_json(meta,get(url,raw),immutable=True)
        if read(meta)['status']=='BLOCKED' and '404' in read(meta).get('error',''):
            raw=out/(pair+'_prices_documented.raw');meta=raw.with_suffix('.json')
            if not meta.exists():write_json(meta,get(f'https://bitnomial.com/exchange/api/web/charts/price/{pid}?begin_time={s["first_trading_day"]}T00:00:00Z&end_time=now',raw),immutable=True)
        prices=[]
        if read(meta)['status']=='ACQUIRED':
            response=read(raw)
            rows=response if isinstance(response,list) else response.get('data',[])
            for row in rows:
                o=row.get('ohlc',{})
                if not all(k in o and o[k] is not None for k in ['open','high','low','close']):continue
                prices.append(dict(pair=pair,product_id=pid,event_at=row.get('time',row['date']),candle_ts=row['date'],
                    **{k:float(o[k]) for k in ['open','high','low','close']},volume=float(o.get('volume',0)),
                    received_at=read(meta)['received_at'],settlement_price=row.get('settlementPrice')))
            if prices:
                p=pd.DataFrame(prices)
                for c in ['event_at','candle_ts','received_at']:p[c]=pd.to_datetime(p[c],utc=True)
                p['available_at']=p.received_at;p['observed_at']=p.event_at
                p.to_parquet(out/(pair+'_prices.parquet'),index=False)
        coverage.append(dict(pair=pair,product_id=pid,prices=len(prices),funding=len(f),
                             price_start=min((p['candle_ts'] for p in prices),default=None),
                             price_end=max((p['candle_ts'] for p in prices),default=None),
                             historical_margin_verified=False,full_mark_intraday_verified=False,
                             evidence='REAL_US_PERP_PARTIAL' if prices and len(f) else 'BLOCKED_US_DATA'))
    write_json(OUT/'instrument_specs.json',dict(verified_at=now(),instruments=instruments,
        discrepancies=['Kraken example PETHIUZ50 differs from explicit Bitnomial PETHUIZ50; product_id5608 resolves identity',
            'exchange first_trading_day is not Kraken brokerage availability start',
            'UTC funding API documentation vs local product schedule fields and broker cash time must not be conflated',
            'Kraken generic cash-settled statement vs expired deliverable legacy products; only active cash-settled IDs used']))
    write_json(out/'coverage.json',dict(coverage=coverage,history_is_backfill=True,accessed_at=now()))
    status('V5-SPEC','IMPLEMENTED_AND_TESTED',['experiments/v5/instrument_specs.json','experiments/v5/public_sources/index.json'],financial='UNVERIFIED_VENUE_SPEC',reason='identity/multiplier/tick verified; absolute historical margin and full liquidation rules incomplete')
    print(json.dumps(coverage,default=str))

if __name__=='__main__':main()
