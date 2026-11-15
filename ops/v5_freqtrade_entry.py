"""V5 zero-fee audit subset. Backtesting only, frozen public markets, network denied."""
import inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
args=sys.argv[1:]
if not args or args[0]!='backtesting' or '--config' not in args or '--fee' not in args:
    raise PermissionError('V5 audit backtesting requires explicit config and fee')
if float(args[args.index('--fee')+1])!=0:raise PermissionError('this launcher only authorizes zero-fee accounting audit')
cfg=json.loads(Path(args[args.index('--config')+1]).read_text(encoding='utf-8'))
if cfg.get('live_enabled') is not False or cfg.get('order_enabled') is not False or cfg.get('trading_mode')!='spot':
    raise PermissionError('offline unlevered accounting subset only')
blocked=[]
def guard(event,argv):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.sendto'}:
        if event=='socket.connect' and any(f.function=='_fallback_socketpair' and f.filename.endswith('socket.py') for f in inspect.stack()):return
        blocked.append(event);raise PermissionError('V5_AUDIT_NO_EXTERNAL_CALLS')
sys.addaudithook(guard)
from freqtrade.exchange import Kraken
from freqtrade.main import main
def markets(self,*args,**kwargs):
    if self._markets:return
    raw=json.loads((ROOT/'experiments/v4/public/kraken_markets.raw').read_text(encoding='utf-8'))['result'];rows=[]
    for key,v in raw.items():
        pair='BTC/USD' if v['wsname']=='XBT/USD' else v['wsname'];base=pair.split('/')[0]
        rows.append(dict(id=key,symbol=pair,base=base,quote='USD',baseId=v['base'],quoteId=v['quote'],active=True,type='spot',spot=True,
            margin=False,swap=False,future=False,option=False,contract=False,linear=None,inverse=None,contractSize=None,settle=None,settleId=None,
            maker=0.,taker=0.,percentage=True,tierBased=False,precision={'amount':10**(-int(v['lot_decimals'])),'price':10**(-int(v['pair_decimals']))},
            limits={'amount':{'min':float(v['ordermin']),'max':None},'price':{'min':None,'max':None},'cost':{'min':float(v.get('costmin',.5)),'max':None},'leverage':{'min':1.,'max':1.}},info=v))
    self._api.set_markets(rows);self._api_async.set_markets(rows);self._markets=self._api.markets
Kraken.reload_markets=markets
if __name__=='__main__':
    code=1
    try:code=main(args) or 0
    except SystemExit as exc:code=exc.code or 0
    finally:
        p=ROOT/'experiments/v5/freqtrade_reconciliation/network_audit.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(dict(exit_code=code,blocked_network_attempts=blocked,network_guard=True,mode='ZERO_FEE_UNLEVERED_FIXTURE_ONLY')),encoding='utf-8')
    sys.exit(code)
