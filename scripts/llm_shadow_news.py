"""Local-only optional news judgment. Missing local model => blocked, no fake output."""
import json,sys,urllib.request
from pathlib import Path
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import write_json,now,digest_file
from quant_platform.research.observation import append_shadow

def main():
    out=ROOT/'experiments/v4/llm_shadow';out.mkdir(exist_ok=True);timestamp=now()
    prompt=ROOT/'configs/llm_shadow_prompt_v4.txt'
    if not prompt.exists():prompt.write_text('Research news judgment only. Return JSON stance (-2 to 2), confidence (0 to 1), and rationale. Never produce trades, weights, prices or orders. Use only supplied contemporaneous news. No historical backfill.\n',encoding='utf-8')
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:11434/api/tags',timeout=3) as r:models=json.load(r).get('models',[])
        if not models:raise RuntimeError('no local model available')
        # A model must be explicitly pinned rather than choosing a changing default.
        pin=ROOT/'configs/llm_shadow_local_pin.json'
        if not pin.exists():raise RuntimeError('local model found but no frozen model/digest pin; retain blocked shadow track')
        raise RuntimeError('no verified contemporaneous news input; no judgment emitted')
    except Exception as exc:
        write_json(out/'metrics.json',dict(schema_version=1,status='BLOCKED',backend='local_ollama',endpoint='127.0.0.1:11434',
               checked_at=timestamp,error=str(exc),prompt_sha256=digest_file(prompt),judgments=0,historical_features=False,order_enabled=False))
        print('BLOCKED_LOCAL_LLM_OR_NEWS; NO_JUDGMENT_EMITTED')
if __name__=='__main__':main()
