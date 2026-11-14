import json
from pathlib import Path
from datetime import datetime,timezone,timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
import pytest
from quant_platform.features.research_imports import import_candidate_file,risk_review_template
from quant_platform.ports.ibkr_research import FutureOptionQuote
ROOT=Path(__file__).resolve().parents[1]

def test_qlib_expression_engine_actual_crosscheck_evidence():
    m=json.loads((ROOT/'experiments/v4/qlib/metrics.json').read_text())
    assert m['status']=='EXPRESSION_CROSSCHECK_PASSED' and len(m['factors'])>=3
    assert all(x['max_absolute_error']<=x['tolerance'] and x['compared_rows']>0 for x in m['factors'])

def test_actual_loopback_and_hashes_verified():
    m=json.loads((ROOT/'experiments/v4/worker/metrics.json').read_text())
    assert m['return_hashes_verified'] and not m['remote_executed'] and m['cpu_gpu_max_absolute_error']<1e-5

def test_candidates_and_static_template_never_enable_trade(tmp_path):
    p=tmp_path/'candidates.json';p.write_text(json.dumps({'schema_version':1,'expressions':['close/lag(close,7)-1']}))
    result=import_candidate_file(p,pd.DataFrame({'close':np.arange(40.)+1}),tmp_path/'ledger.jsonl')
    assert result[0]['status']=='PENDING_HUMAN_APPROVAL'
    t=risk_review_template('BLOCKED','RESEARCH_NEGATIVE');assert not t['order_enabled'] and t['source']=='STATIC_TEMPLATE_NOT_LLM'

def test_ibkr_types_are_only_local_research_contracts():
    t=datetime(2020,1,1,tzinfo=timezone.utc)
    q=FutureOptionQuote('test','underlying',t+timedelta(days=30),Decimal(100),'CALL',Decimal(10),'explicit deliverable',t,t)
    assert q.multiplier==10 and not hasattr(q,'submit_order')

def test_l2_evidence_same_period_and_not_primary():
    m=json.loads((ROOT/'experiments/v4/l2/metrics.json').read_text())
    assert m['terminal_reused'] is False and m['not_comparable_to_tenyear_headline']
    values=m['cost_comparison']
    for cost in ['OPTIMISTIC','BASE','STRESS']:
        a=values[f'ridge_l2_with_dvol/{cost}'];b=values[f'ridge_l2_without_dvol/{cost}']
        assert a['days']==b['days'] and a['start']==b['start'] and a['end']==b['end']
