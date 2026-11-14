import json,zipfile
from pathlib import Path
import pytest
from quant_platform.research.governance import digest_file,Ledger
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/v4'

def test_every_costed_engine_archive_matches_its_recorded_config():
    metrics=json.loads((OUT/'backtest_summary/metrics.json').read_text())
    assert metrics
    for key,item in metrics.items():
        path=ROOT/item['engine_archive'];assert digest_file(path)==item['archive_sha256'],key
        config=json.loads((path.parent/'config.json').read_text())
        with zipfile.ZipFile(path) as z:
            stored=json.loads(z.read(next(n for n in z.namelist() if n.endswith('_config.json'))))
        assert stored['v4_signals']==config['v4_signals'],key
        assert not any(k in stored['exchange'] for k in ['key','secret','password']),key

def test_model_and_data_message_companion_manifests():
    m=json.loads((OUT/'contracts/metrics.json').read_text());assert m['count']>0
    for r in m['artifacts']:
        assert r['schema_version']==1 and digest_file(ROOT/r['path'])==r['sha256']

def test_disabled_observation_template_and_unapproved_candidate():
    c=json.loads((ROOT/'configs/disabled_observation_template.json').read_text())
    assert c['enabled'] is False and c['order_enabled'] is False and c['live_enabled'] is False
    m=json.loads((OUT/'auto_mining/metrics.json').read_text())
    assert not m['auto_mining_enabled'] and not m['financial_evaluation']
    assert m['status']=='PENDING_HUMAN_APPROVAL'

def test_h3_real_data_is_filter_only_and_not_terminal():
    m=json.loads((OUT/'h3/metrics.json').read_text())
    assert not m['derivatives_traded'] and not m['terminal_reused']
    assert m['not_tenyear_L1_evidence']

def test_independent_same_seed_retraining_reproduced():
    m=json.loads((OUT/'reproduction/metrics.json').read_text())
    assert m['passed'] and not m['terminal_evaluated']

def test_engine_has_no_fills_on_missing_canonical_candles():
    import pandas as pd
    bars=pd.read_parquet(OUT/'bars.parquet')
    eligible={('BTC/USD' if pair=='XBTUSD' else 'ETH/USD'):set(g.candle_ts) for pair,g in bars.groupby('pair')}
    metrics=json.loads((OUT/'backtest_summary/metrics.json').read_text())
    for key,item in metrics.items():
        with zipfile.ZipFile(ROOT/item['engine_archive']) as z:
            name=next(n for n in z.namelist() if n.endswith('.json') and not n.endswith('_config.json') and '_V4TargetWeights' not in n)
            trades=json.loads(z.read(name))['strategy']['V4TargetWeights']['trades']
        for trade in trades:
            for order in trade['orders']:
                timestamp=pd.Timestamp(order['order_filled_timestamp'],unit='ms',tz='UTC').floor('D')
                assert timestamp in eligible[trade['pair']],(key,trade['pair'],timestamp)
