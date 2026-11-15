"""Actual installed validator, process-wide network denial, no exchange instance."""
import inspect,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def guard(event,args):
    if event.startswith('socket.') and event not in {'socket.__new__'}: raise PermissionError('OFFLINE_PROBE')
sys.addaudithook(guard)
import freqtrade
from freqtrade.exchange import Kraken
from freqtrade.enums import TradingMode,MarginMode
results=[]
for mode,margin in [(TradingMode.SPOT,MarginMode.NONE),(TradingMode.FUTURES,MarginMode.ISOLATED),(TradingMode.FUTURES,MarginMode.CROSS)]:
    dummy=SimpleNamespace(_supported_trading_mode_margin_pairs=Kraken._supported_trading_mode_margin_pairs,name='Kraken')
    try:
        Kraken.validate_trading_mode_and_margin_mode(dummy,mode,margin); result='SUPPORTED'
    except Exception as exc: result=type(exc).__name__+': '+str(exc)
    results.append(dict(mode=mode.value,margin=margin.value,result=result))
out=ROOT/'experiments/v5/freqtrade_capability';out.mkdir(parents=True,exist_ok=True)
(out/'metrics.json').write_text(json.dumps(dict(version=freqtrade.__version__,network_guard=True,actual_validation_calls=results,
    supported_pairs=[(a.value,b.value) for a,b in Kraken._supported_trading_mode_margin_pairs],
    required_us_semantics=dict(kraken_bitnomial_futures=False,integer_contract_multiplier='NOT_NATIVE_US_VERIFIED',
        fixed_per_contract_fee='NOT_NATIVE_US_VERIFIED',two_wallet_delay='NOT_NATIVE_US_VERIFIED',liquidation_fee='NOT_INCLUDED_BY_DOCUMENTED_ENGINE'),
    decision='V5_EVENT_ENGINE_REQUIRED; FREQTRADE_SPOT_SUBSET_RECONCILIATION_ONLY'),indent=2),encoding='utf-8')
(out/'validator_source.txt').write_text(inspect.getsource(Kraken.validate_trading_mode_and_margin_mode),encoding='utf-8')
print(json.dumps(results))
