"""Truthful ID-level status from real receipts, never infer integration from installed code."""
import json,sys
from pathlib import Path
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import write_json,now,git_sha
OUT=ROOT/'experiments/v4'

def main():
    requirements={}
    def item(rid,status,evidence,receipt_name=None,reason=None):
        recs=sorted((OUT/'commands').glob((receipt_name or 'tests-optional')+'-*.json'),key=lambda p:p.stat().st_mtime)
        if status=='IMPLEMENTED_NOT_RUN' and receipt_name is None:recs=[]
        r=json.loads(recs[-1].read_text()) if recs else {}
        requirements[rid]=dict(status=status,evidence=evidence,command=r.get('command'),exit_code=r.get('exit_code'),
                   receipt=str(recs[-1].relative_to(ROOT)) if recs else None,reason=reason,verified_at=now(),git_sha=git_sha(ROOT))
    tested='IMPLEMENTED_AND_TESTED'
    item('R-MOD-04',tested,['src/quant_platform/models/modalities.py','experiments/v4/tests-optional.xml'])
    item('R-MOD-05',tested,['experiments/v4/calibration/metrics.json'],'calibration')
    item('R-FAC-03',tested,['src/quant_platform/features/mining.py','experiments/v4/tests-optional.xml','experiments/v4/auto_mining/metrics.json'])
    item('R-OSS-01',tested,['docs/OPEN_SOURCE_REUSE_AUDIT.md','THIRD_PARTY_NOTICES.md','experiments/v4/open_source/metrics.json'],'open-source-docs')
    item('R-OSS-02',tested,['experiments/v4/qlib/metrics.json','experiments/v4/tests-integration.xml'],'qlib-offline-smoke',
         'Installed 0.9.7 successfully; requested GitHub 0.9.8 tag audited but distribution unavailable on index. Exact difference disclosed.')
    item('R-OPS-01',tested,['experiments/v4/worker/metrics.json','reports/BLOCKED_REMOTE_GPU.md','ops/jobs/v4-local-loopback/manifest.json'],'worker-loopback',
         'Protocol and local real-weight GPU parity passed; no configured authorized current remote channel, no remote execution claimed.')
    item('R-VAL-04',tested,['experiments/v4/optional_research/metrics.json','experiments/v4/extended/metrics.json'],'ablations-v4')
    item('R-OBS-01',tested,['experiments/v4/forward/metrics.json','experiments/v4/forward/settlement_metrics.json','experiments/v4/forward/input_check_metrics.json','tests/test_v4_forward_snapshots.py'],'forward-append',
         'Failed SIDGTN diagnostic track only; not promoted as an Alpha observation candidate. Labels currently immature.')
    item('R-OBS-02','BLOCKED',['experiments/v4/llm_shadow/metrics.json','scripts/llm_shadow_news.py','experiments/v4/tests-optional.xml'],'local-llm-shadow',
         'Local Ollama/model unavailable; prompt/hash-chain contract and refusal tests exist. No news judgments fabricated.')
    item('R-DATA-05',tested,['experiments/v4/prospective/coverage/metrics.json','experiments/v4/funding_export/whole_metrics.json','experiments/v4/h3/metrics.json'],'prospective-l2',
         'Partial source coverage: DVOL/Fear-Greed/Wikipedia and Kraken funding acquired; BitMEX HTTP403 and historical basis absent, masked.')
    item('R-FAC-04','BLOCKED',['src/quant_platform/data/prospective.py','experiments/v4/public/fomc.receipt.json','experiments/v4/public/cpi.receipt.json','experiments/v4/tests-optional.xml'],
         reason='Current official calendars acquired; historical announced_at not proven. Calendar contract rejects hindsight injection; group disabled, halvings annotation only.')
    item('R-FAC-05',tested,['experiments/v4/l2/metrics.json','experiments/v4/h3/metrics.json'],'h3-funding',
         'DEV-only same-period DVOL and crowding tests; not new terminal evidence and not a ten-year claim.')
    item('R-FAC-06',tested,['experiments/v4/sentiment_exploration/metrics.json'],'prospective-l2',
         'Exploratory lagged proxies only; no LLM historical feature.')
    item('R-MON-01',tested,['experiments/MONITORING_RULES_V4.md','experiments/v4/monitoring/metrics.json','experiments/v4/tests-optional.xml'],'monitor')
    item('R-OSS-03','BLOCKED',['docs/OPEN_SOURCE_REUSE_AUDIT.md','src/quant_platform/features/research_imports.py','experiments/v4/tests-integration.xml'],
         reason='Required Linux/WSL2 + Docker plus local chat/embedding runtime unavailable. No upstream RD-Agent execution claimed.')
    item('R-OSS-04','BLOCKED',['docs/OPEN_SOURCE_REUSE_AUDIT.md','src/quant_platform/features/research_imports.py','experiments/v4/llm_shadow/metrics.json'],
         reason='No usable local Ollama model; static risk-review template is explicitly not an LLM memo or upstream integration.')
    item('R-OSS-05',tested,['experiments/v4/freqai_smoke/metrics.json'],'freqai-smoke',
         'Real FreqAI LightGBM.fit synthetic availability smoke; no online bot and no main-loop integration or financial result.')
    item('R-DATA-04','IMPLEMENTED_NOT_RUN',['src/quant_platform/data/context.py','configs/v4.json','experiments/DECISIONS.md'],
         reason='Default context pool fixed to BTC/ETH per USER_INPUTS. Additional pairs not ingested; context-only contract exists, graph remains two-node contract only.')
    item('R-IBKR-01',tested,['src/quant_platform/ports/ibkr_research.py','experiments/v4/tests-integration.xml'],'integration-tests',
         'Typing-only local snapshot contracts; existing adapters/ibkr remains docs-only. No SDK, account or connection.')
    item('R-DATA-06','BLOCKED',['experiments/v4/public/prediction_market.receipt.json','experiments/v4/public/gdelt.receipt.json','experiments/v4/public/gdelt_retry2.receipt.json','experiments/v4/public/gdelt_retry3.receipt.json'],
         reason='Forward public prediction-market metadata snapshot acquired; GDELT historical/real-time API failed three bounded attempts. No historical backfill or usable probability series claimed.')
    write_json(OUT/'p1/metrics.json',dict(schema_version=1,requirements=requirements,integration='PARTIAL_INTEGRATION',
          projects={'Freqtrade':'ACTUAL_OFFLINE_ENGINE','Qlib':'ACTUAL_THREE_FACTOR_ADAPTER','RD-Agent':'DEFER_BLOCKED','FinRL-X':'REFERENCE_ONLY','TradingAgents':'DEFER_BLOCKED','FreqAI':'ACTUAL_SYNTHETIC_FIT_SMOKE'}))
    blocked='''# Optional blockers and unblocking conditions

Local LLM/news shadow: the local-only endpoint probe failed; no model or judgment is claimed. To unblock, make a free local Ollama model available with a fixed digest, supply contemporaneous public-news snapshots and a pinned local model config, then execute the shadow command. No paid/key-based fallback is permitted. The existing conservative command currently records BLOCKED rather than pretending to run an unpinned model.

RD-Agent: usable Linux/WSL2, Docker and local chat plus embedding runtime are missing. No escalation, firmware, service or security-setting change was performed. File-only candidates and all dangerous-DSL refusals work independently. TradingAgents is similarly deferred until a local-model research invocation can actually run.

Historical event calendars: current FOMC/CPI pages are not proof of historical announced_at. Supply auditable announcement vintages before enabling a historical event factor. No hindsight dates were injected. Halving windows remain annotations, not a statistical finding.

Context assets: default scope remains BTC/ETH. Additional archive members are not ingested under that conservative default; optional context contract exists without activation. Two-node graph tests make no financial performance claim.

GDELT: three bounded anonymous attempts failed. Keep group masked; no mirrors, proxy or VPN. Forward prediction-market metadata is just a timestamped snapshot, not reconstructed historical probability. BitMEX access failed with HTTP403; do not circumvent. Kraken funding archive works, but does not replace absent ten-year BitMEX evidence or historical basis data.

Software remains PARTIAL because conformance gaps and the fixed tail-loss registration deviation are disclosed. No Alpha promotion or capital authority follows from working research code.
'''
    (ROOT/'reports/BLOCKED_OPTIONAL_V4.md').write_text(blocked,encoding='utf-8')
    (ROOT/'reports/BLOCKED_REMOTE_GPU.md').write_text('''# BLOCKED_REMOTE_GPU

ENV_SETUP.md provides only a manual audit workflow, without an authorized host alias, user or remote work directory. Tailscale read-only status succeeded but peers do not identify an authorized workstation for automatic connection. No peer scan, password entry, service or remote configuration change was attempted. The previous task's preserved remote 4090 evidence is historical, not proof of a working current channel.

Local protocol proof: experiments/v4/worker/metrics.json. Bundle: ops/jobs/v4-local-loopback.zip. It contains hashes, minimal code, a real DEV batch and frozen weights; no credentials. On a user-chosen machine with existing NumPy/PyTorch, unpack and run `python worker_entry.py`; copy the generated return/ directory and return_manifest.json back. This is a fixed-batch inference/protocol check, not a remote training claim. Validate hashes with quant_platform.workers.jobs.verify_return. For actual remote training, first supply an explicitly authorized channel and environment; Lenovo already completed training.
''',encoding='utf-8')
    (ROOT/'ops/README_V4.md').write_text('''# V4 offline worker and replay

Package and local loopback: `.venv-training/Scripts/python.exe scripts/v4_worker_loopback.py`. Worker manifest command is strictly `python worker_entry.py`; payload credential patterns, names, traversal and hashes are checked. Return includes frozen weights, inference output, metrics, environment and log. Real remote connection/training was unavailable and is not implied by this test.

Source acquisitions are separately named v4_public_audit, optional_public, prospective_acquire and funding_export/whole scripts. They never run in offline replay. The local LLM script only targets loopback and currently records BLOCKED. No trading command is present in worker manifests. See README_V4.md and reports/BLOCKED_REMOTE_GPU.md.
''',encoding='utf-8')
    print('OPTIONAL_REQUIREMENT_STATUS_AND_BLOCKERS_WRITTEN')
if __name__=='__main__':main()
