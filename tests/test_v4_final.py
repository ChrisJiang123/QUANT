import json,subprocess,sys
from pathlib import Path
import pytest
from quant_platform.research.governance import consume_final,verify_freeze,digest_file
ROOT=Path(__file__).resolve().parents[1]

def test_real_final_second_call_is_refused():
    manifest=json.loads((ROOT/'experiments/v4/frozen.json').read_text());verify_freeze(manifest)
    with pytest.raises(PermissionError,match='ALREADY_CONSUMED'):
        consume_final(ROOT/'experiments/FINAL_CONSUMED.json',manifest)

def test_actual_fresh_holdout_cannot_be_claimed_passed():
    m=json.loads((ROOT/'experiments/v4/FRESH_HOLDOUT/metrics.json').read_text())
    assert m['independent_windows']<12 and m['status']=='INCONCLUSIVE_DATA'

def test_report_numbers_provenance_hashes():
    p=ROOT/'experiments/v4/report_provenance.json';assert p.exists()
    m=json.loads(p.read_text())
    for group in ['sources','reports']:
        for name,sha in m[group].items():assert digest_file(ROOT/name)==sha,name
    assert 'experiments/v4/scorecard/metrics.json' in m['sources']

def test_engine_analysis_semantics_not_only_return_code():
    p=ROOT/'experiments/v4/freqtrade_v2/analyses_v3/metrics.json';m=json.loads(p.read_text())
    assert len(m['commands'])==2
    assert all(r['semantic_pass'] and r['exit_code']==0 for r in m['commands'])
