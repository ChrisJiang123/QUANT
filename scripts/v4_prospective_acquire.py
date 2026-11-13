import json,re
from v4_bootstrap import ROOT
from v4_public_audit import fetch

def main():
    public=ROOT/'experiments/v4/public'
    for page in [2,3]:
        prior=public/('deribit_dvol.raw' if page==2 else f'deribit_dvol_page{page-1}.raw')
        if not prior.exists():break
        data=json.loads(prior.read_text())['result']['data']
        if not data:break
        end=min(r[0] for r in data)-1
        if end<=1609459200000:break
        fetch((f'deribit_dvol_page{page}',f'https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp=1609459200000&end_timestamp={end}&resolution=86400'))
    # Exact official documentation endpoint discovery, not regional circumvention.
    raw=(public/'kraken_funding_docs.raw').read_text(encoding='utf-8')
    urls=list(dict.fromkeys(re.findall(r'https://futures\.kraken\.com[^\s"<>\\]+',raw)))
    print('OFFICIAL_FUNDING_URLS',urls[:5])
    fetch(('kraken_funding_correct','https://futures.kraken.com/derivatives/api/v3/historical-funding-rates?symbol=PF_XBTUSD'))
if __name__=='__main__':main()
