"""Append-only DEV corrections and genuine external-signal truncation checks."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import digest_file, offline_guard, verify_freeze, write_json
from quant_platform.research.rule_forward import DAY, PAIRS, POLICY, historical_signals
import v4_backtest as old
from forward_finance import engine
from v4_engine_metrics import reconciliation

OUT=ROOT/'experiments/next_step_20261113'

def main():
    source=ROOT/'experiments/v4'
    verify_freeze(json.loads((source/'frozen.json').read_text()))
    preserved=json.loads((OUT/'v4_evidence_preservation.json').read_text())
    for p,h in preserved.items(): assert digest_file(ROOT/p)==h,p
    provenance=json.loads((source/'report_provenance.json').read_text())
    for group in ['sources','reports','media']:
        for p,h in provenance[group].items(): assert digest_file(ROOT/p)==h,p
    validation=json.loads((source/'validation/metrics.json').read_text())
    normalizers=validation.get('normalizers',validation.get('folds',[]))
    # Explicitly discover the recorded list; do not silently accept an absent check.
    if not normalizers:
        normalizers=next(v for v in validation.values() if isinstance(v,list) and v and 'train_cutoff' in v[0])
    for n in normalizers:
        assert pd.Timestamp(n['train_label_max'])<pd.Timestamp(n['train_cutoff'])
        assert pd.Timestamp(n['train_cutoff'])<=pd.Timestamp(n['validation_start'])-28*DAY
        assert pd.Timestamp(n['fit_max_decision'])<pd.Timestamp(n['train_cutoff'])
    f=pd.read_parquet(source/'features.parquet')
    dev=f[f.candle_ts<pd.Timestamp('2024-12-31',tz='UTC')].copy()
    audit=OUT/'external_signal_truncation'; audit.mkdir(parents=True,exist_ok=True)
    old_out=old.OUT; old_revision=old.REVISION
    original_full=old.signals('rule',split='DEV')
    cutoff=pd.Timestamp('2023-12-31',tz='UTC')
    try:
        old.OUT=audit/'original_truncated'; old.REVISION='audit'
        old.OUT.mkdir(parents=True,exist_ok=True)
        dev[dev.candle_ts<=cutoff].to_parquet(old.OUT/'features.parquet',index=False)
        truncated=old.signals('rule',split='DEV')
        discrepancies=[]
        for pair in PAIRS:
            name='BTC_USD' if pair=='XBTUSD' else 'ETH_USD'
            a=pd.read_parquet(original_full/(name+'.parquet')).set_index('candle_ts')
            b=pd.read_parquet(truncated/(name+'.parquet')).set_index('candle_ts')
            for field in ['target_weight','decision','gap_mask']:
                z=a.loc[b.index,field]!=b[field]
                discrepancies.append(dict(pair=pair,field=field,differences=int(z.sum()),
                                           dates=[str(x) for x in z[z].index]))
    finally: old.OUT=old_out; old.REVISION=old_revision
    with offline_guard():
        fixed=historical_signals(dev)
        checks=[]
        for t in [pd.Timestamp('2018-01-11',tz='UTC'),cutoff]:
            cut=historical_signals(dev[dev.candle_ts<=t])
            for p in PAIRS:
                pd.testing.assert_frame_equal(fixed[p][fixed[p].candle_ts<=t].reset_index(drop=True),cut[p])
                checks.append(dict(pair=p,cutoff=str(t),equal=True))
    signal_dir=OUT/'causal_dev_correction/signals'; signal_dir.mkdir(parents=True,exist_ok=True)
    original_diff=[]
    for pair,s in fixed.items():
        name='BTC_USD' if pair=='XBTUSD' else 'ETH_USD'
        original=pd.read_parquet(original_full/(name+'.parquet')).set_index('candle_ts')
        new=s.set_index('candle_ts')
        mask=original.loc[new.index,'gap_mask']!=new.gap_mask
        original_diff.append(dict(pair=pair,changed_gap_flags=int(mask.sum()),
                                  changed_weights=int((original.loc[new.index,'target_weight']!=new.target_weight).sum()),
                                  changed_decisions=int((original.loc[new.index,'decision']!=new.decision).sum()),
                                  dates=[str(t) for t in mask[mask].index]))
        s['model_version']='rule-receipt-v1-causal-dev-audit'
        s.to_parquet(signal_dir/(name+'.parquet'),index=False)
    write_json(audit/'metrics.json',dict(original_truncated=discrepancies,causal_truncated=checks,
               dev_signal_differences=original_diff,original_status='FAIL_FUTURE_ADJACENCY_GAP_MASK',
               corrected_status='PASS',normalizer_folds_checked=len(normalizers),
               old_report_provenance_verified=True,final_rerun=False))
    alias_cut=pd.Timestamp('2017-10-01',tz='UTC')
    try:
        old.OUT=audit/'original_alias_truncated'; old.REVISION='audit'
        old.OUT.mkdir(parents=True,exist_ok=True)
        dev[dev.candle_ts<=alias_cut].to_parquet(old.OUT/'features.parquet',index=False)
        alias_path=old.signals('rule',split='DEV')
        alias=[]
        for pair in PAIRS:
            name='BTC_USD' if pair=='XBTUSD' else 'ETH_USD'
            a=pd.read_parquet(original_full/(name+'.parquet')).set_index('candle_ts')
            b=pd.read_parquet(alias_path/(name+'.parquet')).set_index('candle_ts')
            changed=(a.loc[b.index,'target_weight']-b.target_weight).abs()>1e-12
            alias.append(dict(pair=pair,changed_weights=int(changed.sum()),
                              original_full_hash=digest_file(original_full/(name+'.parquet')),
                              actual_truncated_hash=digest_file(alias_path/(name+'.parquet'))))
        assert all(a['changed_weights']>0 for a in alias)
        cut=historical_signals(dev[dev.candle_ts<=alias_cut])
        for pair in PAIRS:
            pd.testing.assert_frame_equal(fixed[pair].loc[fixed[pair].candle_ts<=alias_cut,cut[pair].columns].reset_index(drop=True),cut[pair])
        write_json(audit/'alias_audit.json',dict(cutoff=str(alias_cut),original=alias,
                   cause='band reuses previous weight array; holding expiry mutates prior day',
                   original_status='FAIL',corrected_status='PASS',
                   affected_weights_executable=False,reason='changed prior-day rows have decision=false',
                   corrected_DEV_engine_results='three-cost metrics and reconciliation preserved separately'))
    finally: old.OUT=old_out; old.REVISION=old_revision
    corrections={}
    for cost in POLICY['cost_per_side']:
        folder=OUT/'causal_dev_correction'/cost
        if (folder/'metrics.json').exists():
            corrections[cost]=json.loads((folder/'metrics.json').read_text()); continue
        cfg=old.configuration('rule',signal_dir,folder/'weight_audit.jsonl')
        m,_,fills=engine(cfg,folder,source/'freqtrade_v2/data',signal_dir,cost,'20161101','20250101')
        rec=reconciliation(folder)
        assert rec['passed'] and not rec['violations']
        if not fills.empty:
            bars=pd.read_parquet(source/'bars.parquet')
            for _,fill in fills.iterrows():
                pair='XBTUSD' if fill.pair=='BTC/USD' else 'ETHUSD'
                at=pd.to_datetime(fill.order_filled_timestamp,unit='ms',utc=True)
                assert ((bars.pair==pair)&(bars.candle_ts==at)).any(),'fill on missing actual candle'
        corrections[cost]=m
    write_json(OUT/'causal_dev_correction/metrics.json',dict(run_id='CAUSAL_DEV_CORRECTION_20261113',
               linked_original='experiments/v4/freqtrade_v2/DEV/rule',segment='DEV_ONLY_EXPLORATORY',
               final_rerun=False,training_started=False,metrics=corrections))
    for p,h in preserved.items(): assert digest_file(ROOT/p)==h,p
    print('PRESERVATION_BOUNDARIES_TRUNCATION_AND_DEV_CORRECTION_PASS')

if __name__=='__main__': main()
