"""Independent actual Freqtrade zero-fee subset against the single V5 book."""
import io,json,re,subprocess,sys,zipfile,shutil
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,status
from quant_platform.v5.instruments import InstrumentSpec
from quant_platform.v5.book import Book
from quant_platform.research.governance import write_json,digest_file
from v4_backtest import configuration

def main():
    folder=OUT/'freqtrade_reconciliation';folder.mkdir(exist_ok=True);data=folder/'data';data.mkdir(exist_ok=True)
    times=pd.date_range('2020-01-01',periods=12,tz='UTC');bars={}
    for pair,offset in [('BTC/USD',100.),('ETH/USD',50.)]:
        p=offset+np.arange(12)*2
        f=pd.DataFrame(dict(date=times,open=p,high=p+1,low=p-1,close=p+.5,volume=100000.));f.to_feather(data/(pair.replace('/','_')+'-1d.feather'));bars[pair]=f.set_index('date')
    cfg=configuration('rule',folder/'unused_signals',folder/'unused_audit')
    cfg.update(strategy='V5FixedLots',v4_data_dir=str(data));write_json(folder/'config.json',cfg)
    command=[str(ROOT/'.venv-freqtrade/Scripts/python.exe'),str(ROOT/'ops/v5_freqtrade_entry.py'),'backtesting',
        '--config',str(folder/'config.json'),'--strategy','V5FixedLots','--strategy-path',str(ROOT/'freqtrade_userdir/v5_audit'),
        '--datadir',str(data),'--timerange','20200101-20200113','--fee','0','--export','trades','--cache','none']
    if not (folder/'receipt.json').exists():
        with (folder/'engine.log').open('xb') as f:p=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        write_json(folder/'receipt.json',dict(command=command,exit_code=p.returncode,log_sha256=digest_file(folder/'engine.log')))
    receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['command']==command and receipt['log_sha256']==digest_file(folder/'engine.log')
    if receipt['exit_code']:raise RuntimeError('Freqtrade subset failed')
    log=(folder/'engine.log').read_text(encoding='utf-8',errors='replace')
    name=re.findall(r'backtest-result-[\d_-]+\.meta\.json',log)[-1].replace('.meta.json','.zip')
    if not (folder/'engine_result.zip').exists():shutil.copy2(ROOT/'freqtrade_userdir/backtest_results'/name,folder/'engine_result.zip')
    with zipfile.ZipFile(folder/'engine_result.zip') as z:
        result=json.loads(z.read(next(n for n in z.namelist() if n.endswith('.json') and not n.endswith('_config.json'))))['strategy']['V5FixedLots']
        wallet=pd.read_feather(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('_wallet.feather')))))
    specs={'BTC/USD':InstrumentSpec('audit:BTC','BTC/USD',.01,.01,fixed_fee=0,initial_margin_fraction=1.),'ETH/USD':InstrumentSpec('audit:ETH','ETH/USD',.5,.01,fixed_fee=0,initial_margin_fraction=1.)}
    book=Book(specs,initial=10000,perp_bps=0,spot_bps=0,wallet_fraction=1.)
    fills=[]
    for trade in result['trades']:
        for o in trade['orders']:fills.append(dict(pair=trade['pair'],**o))
    fills=pd.DataFrame(fills);assert len(fills)==4
    fills['at']=pd.to_datetime(fills.order_filled_timestamp,unit='ms',utc=True)
    quotes=wallet.groupby('date').total_quote.sum();comparison=[]
    for date in sorted(set(quotes.index)|set(fills['at'])):
        prices={pair:float(f.loc[date,'open']) for pair,f in bars.items()};book.mark_to(prices)
        if date in quotes.index:comparison.append(dict(at=str(date),freqtrade=float(quotes.loc[date]),v5=book.equity(prices),error=book.equity(prices)-float(quotes.loc[date])))
        for _,o in fills[fills['at']==date].iterrows():
            n=round(float(o.amount)/specs[o.pair].multiplier)
            assert abs(n*specs[o.pair].multiplier-float(o.amount))<1e-8
            current=book.virtual_position('audit',o.pair).quantity
            target=current+n if o.ft_order_side=='buy' else current-n
            assert float(o.safe_price)==prices[o.pair]
            book.sync({('audit',o.pair):target},prices,date,'freqtrade_accounting_subset')
    final_error=book.equity()-float(result['final_balance'])
    write_json(folder/'metrics.json',dict(fixture_only=True,financial_evidence=False,engine='Freqtrade2026.10 vs V5Book',
        trades=len(result['trades']),fills=len(fills),zero_fee=True,unlevered_notional=True,funding=0,
        final_error=final_error,max_wallet_error=max(abs(r['error']) for r in comparison),comparison=comparison,
        source_engine_hash=digest_file(folder/'engine_result.zip')))
    assert abs(final_error)<1e-6
    assert max(abs(r['error']) for r in comparison)<1e-6
    status('V5-ENGINE','IMPLEMENTED_AND_TESTED',['experiments/v5/tests-engine-first.xml','experiments/v5/freqtrade_reconciliation/metrics.json'],financial='SCENARIO_ENGINE_ONLY_NOT_US_VENUE_VALIDATION')
    print('ACTUAL_FREQTRADE_ZERO_FEE_LEDGER_RECONCILIATION_PASS')

if __name__=='__main__':main()
