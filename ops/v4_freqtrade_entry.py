"""Isolated Freqtrade CLI using public market snapshot, with process-wide network denial.

This launcher replaces only market metadata loading, never the backtest engine.
No exchange network request, bot startup, credentials, or database connection.
"""
import json
import sys
import inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.adapters.crypto.offline_freqtrade import authorize_cli
from quant_platform.research.governance import ensure_research
authorize_cli(sys.argv[1:])
blocked=[]
def deny_network(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.sendto'}:
        # Windows implements socketpair with a loopback self-pipe. This exact stdlib
        # call is event-loop IPC, not an exchange or external connection.
        if event=='socket.connect' and any(f.function=='_fallback_socketpair' and f.filename.endswith('socket.py') for f in inspect.stack()):
            return
        blocked.append(event)
        raise PermissionError('V4_OFFLINE_NETWORK_DENIED: '+event)
sys.addaudithook(deny_network)

from freqtrade.resolvers import ExchangeResolver
from freqtrade.exchange import Kraken
from freqtrade.main import main

original_reload=Kraken.reload_markets
def snapshot_reload(self,*args,**kwargs):
    if self._markets:return
    raw=json.loads((ROOT/'experiments/v4/public/kraken_markets.raw').read_text())['result']
    markets=[]
    for key,value in raw.items():
        pair='BTC/USD' if value['wsname']=='XBT/USD' else value['wsname']
        base=pair.split('/')[0]
        markets.append(dict(id=key,symbol=pair,base=base,quote='USD',baseId=value['base'],quoteId=value['quote'],
                            active=True,type='spot',spot=True,margin=False,swap=False,future=False,option=False,contract=False,
                            linear=None,inverse=None,contractSize=None,settle=None,settleId=None,
                            maker=.008,taker=.008,percentage=True,tierBased=True,
                            precision={'amount':10**(-int(value['lot_decimals'])),'price':10**(-int(value['pair_decimals']))},
                            limits={'amount':{'min':float(value['ordermin']),'max':None},'price':{'min':None,'max':None},
                                    'cost':{'min':float(value.get('costmin',.5)),'max':None},'leverage':{'min':1.,'max':1.}},info=value))
    self._api.set_markets(markets);self._api_async.set_markets(markets)
    self._markets=self._api.markets
Kraken.reload_markets=snapshot_reload

if __name__=='__main__':
    try: main(sys.argv[1:])
    except SystemExit as exc: code=exc.code or 0
    folder=ROOT/'experiments/v4/freqtrade'
    folder.mkdir(exist_ok=True)
    with (folder/'network_audit.jsonl').open('a') as f:
        f.write(json.dumps({'schema_version':1,'argv':sys.argv[1:],'exit_code':code,'blocked_network_attempts':blocked,
                            'network_guard_active':True,'engine':'freqtrade','metadata_source':'public REST frozen snapshot',
                            'container_network_none':False})+'\n')
    sys.exit(code)
