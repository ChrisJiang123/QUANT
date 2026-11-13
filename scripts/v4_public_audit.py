"""Separate, anonymous public acquisition step; never executed by offline pipeline."""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from v4_bootstrap import ROOT, run

OUT = ROOT/'experiments/v4/public'
OUT.mkdir(parents=True, exist_ok=True)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
URLS = {
 'kraken_archive': 'https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data',
 'kraken_fees': 'https://www.kraken.com/features/fee-schedule',
 'kraken_pro_fees': 'https://pro.kraken.com/fees',
 'kraken_checksums': 'https://assets.kraken.com/marketing/institutions/OHLCVT_Full_PARTS_SHA256SUMS.txt',
 'kraken_q3_head': 'https://assets.kraken.com/marketing/institutions/Kraken_OHLCVT_2026Q3.zip',
 'kraken_rest_btc': 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1440',
 'kraken_rest_eth': 'https://api.kraken.com/0/public/OHLC?pair=ETHUSD&interval=1440',
 'kraken_markets': 'https://api.kraken.com/0/public/AssetPairs?pair=XBTUSD,ETHUSD',
 'bitmex_funding': 'https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&reverse=true',
 'kraken_futures': 'https://futures.kraken.com/derivatives/api/v3/historicalfundingrates?symbol=PF_XBTUSD',
 'deribit_dvol': 'https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp=1609459200000&end_timestamp=1794441600000&resolution=86400',
 'fear_greed': 'https://api.alternative.me/fng/?limit=0&format=json',
 'wikipedia': 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/Bitcoin/daily/2016110100/2026111100',
 'fomc': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
 'cpi': 'https://www.bls.gov/schedule/news_release/cpi.htm',
 'gdelt': 'https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin&mode=artlist&format=json&maxrecords=10',
 'freqtrade_pypi':'https://pypi.org/pypi/freqtrade/2026.10/json',
}
REPOS={'freqtrade':'freqtrade/freqtrade','qlib':'microsoft/qlib','rdagent':'microsoft/RD-Agent',
       'finrl':'AI4Finance-Foundation/FinRL-Trading','tradingagents':'TauricResearch/TradingAgents'}
for short, repo in REPOS.items():
    for kind, suffix in [('repo',''),('release','/releases/latest'),('license','/license'),('tags','/tags?per_page=5')]:
        URLS[f'{short}_{kind}']=f'https://api.github.com/repos/{repo}{suffix}'

def fetch(item):
    name,url=item
    receipt=OUT/f'{name}.receipt.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    result=dict(schema_version=1, url=url, collected_at=datetime.now(timezone.utc).isoformat(),
                anonymous=True, proxy_used=False)
    try:
        headers={'User-Agent':'QuantSIDGTN-Research/4.4 (anonymous public audit)'}
        if name=='kraken_q3_head': headers['Range']='bytes=0-0'
        with OPENER.open(urllib.request.Request(url,headers=headers),timeout=30) as resp:
            data=resp.read(32*1024*1024+1)
            if len(data)>32*1024*1024: raise ValueError('bounded acquisition limit exceeded')
            result.update(status='ACQUIRED', http_status=resp.status, headers=dict(resp.headers),
                          sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        path=OUT/f'{name}.raw'
        with path.open('xb') as f: f.write(data)
        os.chmod(path,0o444)
        result['path']=path.relative_to(ROOT).as_posix()
    except Exception as exc:
        result.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name,result['status'],flush=True)
    return result

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(fetch,URLS.items()))
    (OUT/'metrics.json').write_text(json.dumps({'schema_version':1,'sources':results},indent=2),encoding='utf-8')
    run('tailscale-readonly',[r'C:\Program Files\Tailscale\tailscale.exe','status','--json'],timeout=20)
    run('wsl-status',['wsl','--status'],timeout=20)
    run('docker-version',['docker','version'],timeout=20)
    run('ollama-local',['curl.exe','--noproxy','*','--max-time','5','http://127.0.0.1:11434/api/tags'],timeout=10)
