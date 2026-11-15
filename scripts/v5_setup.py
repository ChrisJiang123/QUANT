"""Preserve V4, initialize V5 state, preregister before any strategy comparison."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import Ledger,digest_file,git_sha,now,verify_freeze,write_json

IDS=['SPEC','DATA','GOV','ENGINE','TREND','CARRY','TACTICAL','RISK','DIAG','CORE','FACTOR','GRAPH','SENTIMENT','FUSION','GPU','VAL','OBS','TEST']

def main():
    out=ROOT/'experiments/v5'; out.mkdir(parents=True,exist_ok=True)
    verify_freeze(json.loads((ROOT/'experiments/v4/frozen.json').read_text(encoding='utf-8')))
    if not (out/'v4_preservation.json').exists():
        files=[ROOT/'experiments/FINAL_CONSUMED.json',ROOT/'experiments/SPLITS_V4.json',ROOT/'experiments/PREREGISTRATION_V4.md',ROOT/'experiments/v4/frozen.json']
        files+=list((ROOT/'reports').glob('*V4*'))
        write_json(out/'v4_preservation.json',dict(source_head=git_sha(ROOT),files={str(p.relative_to(ROOT)):digest_file(p) for p in files if p.is_file()}),immutable=True)
    cfg=json.loads((ROOT/'configs/v5/research.json').read_text(encoding='utf-8'))
    splits=dict(DEV=dict(start='2016-11-01',end_exclusive='2025-01-01',evidence='PREVIOUSLY_SEEN_DEVELOPMENT'),
        validation_years=cfg['walk_forward_years'],purge_days=28,embargo_days=28,
        terminal=dict(status='INCONCLUSIVE_DATA',clean_windows=0,execution='REFUSED',reason='V4 final already consumed; tool change does not clean prices'),
        forward=dict(start='V5_MODEL_FREEZE_TIMESTAMP',required_independent_28d_windows=12))
    write_json(out/'SPLITS_V5.json',splits,immutable=True)
    hypotheses='''# V5 hypotheses — registered before comparison
H1 long trend vs equal-risk perpetual buy/hold; H1s asymmetric shorts vs long-only.
H2 realized/EWMA vs SIDGTN annual volatility under identical risk/cost inputs.
H3 carry only if 7d known annual rate>=8% and 28d complete net costs covered; compare isolated curve/correlation, 30d sensitivity separate.
H3c funding rank>90% reduces only trend longs; H4 announced-at event window reduces directional books only (BLOCKED if historical publication unproved).
H5 >3 predicted daily sigma drop + known funding rank high and >50% retreat; isolated max3d tactical, main disabled.
SIDGTN core/factor/graph each five seeds, same snapshot and folds; sentiment real-days>=365 before enabled financial experiment; no synthetic financial claims.
Failures and zero-trade strategies are results, not permission to tune. Each family ONE configuration; caps30/family,12/decision remain ceilings.
'''
    hp=out/'HYPOTHESES_V5.md'
    if not hp.exists(): hp.write_text(hypotheses,encoding='utf-8')
    assumption=out/'ASSUMPTIONS.md'
    if not assumption.exists(): assumption.write_text('# Research assumptions\n\nFrozen defaults are configs/v5/research.json. USD7000 is a simulated balance, not real funds or an FX conversion. 25% scale, 15/25% breakers, .60/.25/.15 budgets,10x cap, asymmetric shorts, unknown margin scenarios, daily adverse-first paths and late historical availability are assumptions. No U.S. execution/survival claim without complete venue evidence. Carry has no assumed free limit fills. Tactical main disabled.\n',encoding='utf-8')
    prereg='''# V5 preregistration
Registered before strategy comparisons and V5 training. Code archive/hash frozen again before first fit; earlier data-quality/environment probes are not validation.
Five horizons1/3/7/14/28; primary14/28. Seeds11/22/33/44/55; variants core→factor→factor+graph. Annual WF2022/2023/2024; max28d purge/embargo, inner stop180d; scalers and direction temperature fit inner/calibration only. All seeds retained. Targets executable-open(t+2) to open(t+2+h), not an already-past close. Historical receipt assumption close+6h; earliest execution next UTC opening. Real forward uses actual response receipt.
Input price60d; primary8 factors; auxiliary funding/rank/change/basis/DVOL/fear-greed/Wikipedia/event masks;8 fixed graph nodes; sentiment masked unless auditable archive is available. Tail q05 pinball explicitly weight .1, BCE .1, vol normalized Huber .1, quantile1.0; horizon weights .1/.1/.1/.35/.35. Return normalization causal annual vol*sqrt(h/365); annual vol floor.10. Train anchors7d; overlapping evaluation retains HAC h-1 and paired portfolio blocks28d. No independent-asset double count.
G1 BOTH primary MAE skill vs zero>0, DM one-sided HAC h-1 p<.05 with Holm across variants/horizons; shorter horizons registered secondary. G2 BASE Sharpe delta>=.20 vs strongest DEV simple baseline at identical cost/risk; paired28d block90% lower>0; DSR>=.95 with all actual trials counted. G3 positiveSTRESS delta, medianseed passesG2, worstseed delta>=0, majority regime segments>=0. G4 >=12 clean independent28d forward windows and consistent direction; currently no eligible clean final so INCONCLUSIVE. Any G1-G3 fail means NO_GO_ALPHA. Survival gate: any strategy account liquidation or >1hardbreaker means NO_GO; fabricated boundary unit tests excluded. SCENARIO_ONLY cannot validate U.S. survival.
Only one source/config and decision mapping per registered family. Alternative direction-logit/median/calibrated mappings are registered diagnostics, not a search for a winning final. Dynamic fusion stays disabled unless registered paired primary/financial increment gate passes; optional later5seed run uses same DEV only.
Costs2/5/15bps perp including fixed per-contract minimum; spot35/85/120bps. STRESS physical-net funding: positive long paymentsx1.5, positive short receiptsx.5; negative funding symmetrically worsened for payer/receiver; virtual allocation must conserve net cash. Path OPEN-funding-adverse-favorable-CLOSE, liquidate before stop on ambiguous segment; favorable-first sensitivity reports bounds separately and cannot select primary. Historical mark unknown=spot SCENARIO_ONLY, missing funding masks/coverage never relabelled as a real perpetual history. Initial/maintenance margin scenarios20%notional/half thereof are NOT verified U.S. margins.
Carry28d expected complete-cost gate, optional30d sensitivity; integer contracts, matchedspotqty; actual-net fees allocated proportionally to virtual changes; no limit-fill benefit,24h internal transfer delay, no external cash injection. Trend max28d, tacticalmax3d, stop2.5daily sigma,3dcooldown,20%band waived for risk/expiry/flip,15/25%breakers with hard lock. Model/parameter changes require new track and new clean future observations.
'''
    pp=out/'PREREGISTRATION_V5.md'
    if not pp.exists(): pp.write_text(prereg,encoding='utf-8')
    regpaths=[pp,hp,assumption,out/'SPLITS_V5.json',ROOT/'configs/v5/research.json',ROOT/'docs/V5_IMPLEMENTATION_DECISIONS.md']
    registration=dict(registered_at=now(),source_head=git_sha(ROOT),files={str(p.relative_to(ROOT)):digest_file(p) for p in regpaths})
    if not (out/'registration.json').exists(): write_json(out/'registration.json',registration,immutable=True)
    state=out/'BUILD_STATE.json'
    if not state.exists():
        write_json(state,dict(schema_version=5,created_at=now(),phase='IMPLEMENTING',requirements={
            'V5-'+i:dict(status='PENDING',implementation='NOT_STARTED',run='NOT_RUN',financial_evidence='NOT_EVALUATED',evidence=[]) for i in IDS}))
        Ledger(out/'ledger.jsonl').append(dict(kind='PREREGISTERED',registration_hash=digest_file(out/'registration.json'),git_sha=git_sha(ROOT)))
    print('V5_REGISTERED_V4_PRESERVED')

if __name__=='__main__': main()
