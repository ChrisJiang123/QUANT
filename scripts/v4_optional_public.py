import json,sys
from v4_public_audit import fetch,REPOS
from v4_bootstrap import ROOT,run
TAGS={'freqtrade':'2026.10','qlib':'v0.9.8','rdagent':'v1.0.0','finrl':'v1.0.0','tradingagents':'v0.7.1'}

def main():
    for name,repo in REPOS.items():
        tag=TAGS[name]
        run(name+'-tag-resolve',['git','ls-remote',f'https://github.com/{repo}.git',f'refs/tags/{tag}',f'refs/tags/{tag}^{{}}'],timeout=30)
        for file in ['README.md','pyproject.toml','LICENSE','NOTICE']:
            fetch((name+'_'+file.replace('.','_'),f'https://raw.githubusercontent.com/{repo}/{tag}/{file}'))
    fetch(('kraken_funding_docs','https://docs.kraken.com/api/docs/futures-api/trading/historical-funding-rates/'))
    fetch(('kraken_funding_export_docs','https://support.kraken.com/articles/export-historical-funding-rates'))
    fetch(('prediction_market','https://gamma-api.polymarket.com/markets?limit=20&active=true&closed=false'))
    # Repeated failures stay blocked; no mirrors, proxies or credentials.
    fetch(('gdelt_retry2','https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin&mode=timelinevolraw&format=json&startdatetime=20240101000000&enddatetime=20241231000000'))
    fetch(('gdelt_retry3','https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin&mode=artlist&format=json&maxrecords=10&timespan=1d'))
if __name__=='__main__':main()
