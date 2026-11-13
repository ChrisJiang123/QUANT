import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_platform.adapters.crypto.offline_freqtrade import run_offline
from quant_platform.research.governance import write_json
from v4_bootstrap import ROOT

def main():
    out=ROOT/'experiments/v4/freqtrade_v2/analyses_v3';out.mkdir(exist_ok=True)
    config=json.loads((ROOT/'experiments/v4/freqtrade_v2/DEV/rule/BASE/config.json').read_text());config['fee']=.0085
    write_json(out/'config.json',config)
    base=['--config',str(out/'config.json'),
          '--strategy','V4AnalysisWeights','--strategy-path',str(ROOT/'freqtrade_userdir/strategies'),
          '--datadir',str(ROOT/'experiments/v4/freqtrade_v2/data'),'--timerange','20220101-20250101']
    results=[]
    for command,extra in [('lookahead-analysis',['--fee','0.0085','--minimum-trade-amount','5','--targeted-trade-amount','10',
                       '--lookahead-analysis-exportfilename',str(out/'lookahead.csv')]),
                         ('recursive-analysis',['--startup-candle','120','240','480'])]:
        result=run_offline(ROOT,[command,*base,*extra],out/(command+'.log'))
        log=(out/(command+'.log')).read_text(encoding='utf-8',errors='replace')
        result['semantic_pass']=result['exit_code']==0 and 'error:' not in log.lower() and 'fatal exception' not in log.lower()
        results.append(result)
        print(command,result['exit_code'],flush=True)
    write_json(out/'metrics.json',{'schema_version':1,'commands':results,'external_signal_limitation':'built-in analyses cannot inspect construction of external predictions; independent truncation and injection tests passed separately'})
if __name__=='__main__':main()
