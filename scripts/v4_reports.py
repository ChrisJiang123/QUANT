"""Generate all numeric research reports from retained metrics and command receipts."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quant_platform.research.governance import write_json,digest_file,now,git_sha
from v4_bootstrap import ROOT
OUT=ROOT/'experiments/v4'; REPORTS=ROOT/'reports'

def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else ({} if default is None else default)
def pct(x):return 'N/A' if x is None else f'{x:.2%}'
def num(x):return 'N/A' if x is None else f'{x:.4f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)+'\n'
def md(name,text):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')

def test_metrics():
    commands=[read(str(p.relative_to(ROOT))) for p in sorted((OUT/'commands').glob('*.json'))]
    runs=[]
    for p in sorted(OUT.glob('*.xml')):
        root=ET.parse(p).getroot();suites=root.findall('testsuite') if root.tag=='testsuites' else [root]
        runs.append(dict(path=str(p.relative_to(ROOT)),modified_at=p.stat().st_mtime,tests=sum(int(s.get('tests',0)) for s in suites),
                    failures=sum(int(s.get('failures',0)) for s in suites),errors=sum(int(s.get('errors',0)) for s in suites),
                    skipped=sum(int(s.get('skipped',0)) for s in suites)))
    write_json(OUT/'tests/metrics.json',dict(schema_version=1,commands=commands,runs=runs))
    persisted=read('experiments/v4/tests/metrics.json')
    return persisted['commands'],persisted['runs']

def docs():
    lineage=read('experiments/v4/data_lineage.json');reg=read('experiments/v4/factor_registry.json',[])
    md('README_V4.md', r'''# QUANT-SIDGTN V4 research system

Use the existing project directory and retained snapshots. No accounts, credentials, orders, leverage or online dry-run bot.

```powershell
.\.venv-training\Scripts\python.exe scripts/run_v4_research.py --offline --no-external-calls
```

The command resumes verified training folds, reuses completed offline engine runs, regenerates metrics/reports and never silently reconsumes final data. Use `--verify-only` for a small frozen-integrity/dry-check replay. Network acquisition is a separate explicitly invoked `scripts/v4_public_audit.py`; the research replay never downloads data.

Forward observations use new snapshots, without rewriting frozen research data. Run `scripts/collect_forward_snapshot.py` as a separate anonymous online acquisition step; it prints the new immutable `data/forward_public/<id>/bars.parquet` path. Then run `scripts/append_forward_prediction.py --snapshot <path>` and `scripts/settle_forward_predictions.py --snapshot <path>`. Both operate on file-only ledgers. A repeated UTC daily prediction is refused/reused, immature labels remain pending, and `--dry-check` on the prediction command verifies inputs/inference without appending. The failed model stays a diagnostic track, not an Alpha candidate.

Core environment: retained CUDA training venv plus pinned analytics packages; original minimal data venv preserved. Freqtrade uses an isolated native venv and CLI-only boundary. Exact inventories: requirements-v4-core-lock.txt and requirements-v4-freqtrade-lock.txt. Docker/WSL virtualization is unavailable; no system security or driver change was made. Native network guard is tested; Docker network-none is not verified.

Start with reports/V4_COMPLETION_REPORT.md and reports/V4_STATUS.json. Original reports/data and the negative old Transformer result remain intact. Fresh holdout is too short for an independent confirmation. New Alpha requires a new preregistration and forward track, never tuning the consumed holdout.
''')
    md('docs/ARCHITECTURE_V4.md','''# Research architecture

Immutable TASK001 + REST_PROVISIONAL snapshots → calendar-gap masks → PIT factors → frozen hypotheses/splits/preregistration → purged expanding annual training → distributions → long/cash target weights → historical exposure/liquidity-proxy gate → external Freqtrade offline CLI → engine wallet/fill reports.

Core imports no trading SDK. External strategy files live in freqtrade_userdir. A separate entry process replaces network market-metadata loading with the anonymous public AssetPairs snapshot, then denies network calls. Standard-library socketpair IPC on Windows is the only narrowly identified loopback exception. No trading command is accepted.

All retained original artifacts have an input ZIP backup and file hashes. New ledger writes use exclusive locking, append-only JSONL and hash links. Per-fold checkpoints resume optimizer/RNG state. Freeze checks model, code, config, source snapshots, hypotheses and preregistration. Consumption markers are claimed before terminal computation, conservatively refusing retries unless explicitly labelled audit reruns.

The final file-hash manifest identifies evaluation dependencies. Later optional modules do not alter the frozen prediction path. Current integration/environment gaps are in the status report; contracts do not imply installed or working third-party software.
''')
    md('docs/MODEL_SPEC_V4.md','''# Model specification

Price tensor [batch, lookback, feature_dim] uses log close return, range/close and log volume. Each asset is aligned to its own UTC calendar; absent bars remain absent values, not forward-filled prices. Asset IDs use sorted ETHUSD, XBTUSD order. Primary factors use the registered trailing windows and missing masks; incomplete primary or price windows are excluded.

Small convolutional price encoder, gated factor MLP, asset embedding, softmax auditable fusion, return quantiles, direction logits, realized-volatility and downside heads. Batch lookback, hidden size, dropout, AdamW decay, search budget, seeds and losses are in configs/v4.json and the immutable preregistration. Targets are raw simple returns; sample weights are uniform. Actual source includes a disclosed auxiliary tail loss omitted from the preregistered JSON; no promotion from this experiment is allowed. See PREREGISTRATION_IMPLEMENTATION_NOTES_V4.md.

Normalization is fitted on inner training only; inner early stopping has its own label purge. Outer annual validation is never used for early stopping. Per-fold parameter count, actual GPU, epoch losses, runtime, memory and inference timings are in experiments/v4/training/**/metrics.json and training.jsonl. Held-out model weights use only DEV mature labels. They are separately hash-frozen before touched/fresh evaluation. Additional modality contracts are not evidence of multimodal Alpha.
''')
    md('docs/FACTOR_CATALOG_V4.md','# Factor registry\n\n'+table(['Name','Formula','Lookback','Hypothesis','Primary'],[[x['name'],x['formula'],x['lookback'],x['hypothesis'],x['primary']] for x in reg])+'''Primary quality evidence: experiments/v4/factor_quality/metrics.json. Cross-asset relative strength/correlation/beta are registered exploratory candidates and are not present in the primary model. Missing, extreme, correlation and annual stability summaries are descriptive, not two-asset cross-sectional IC evidence. LLM outputs never enter this registry's historical values.
''')
    md('docs/DATA_LINEAGE_V4.md','# Data lineage\n\n'+json.dumps(lineage,ensure_ascii=False,indent=2)+'''\n
Original canonical Parquet files and raw content hashes remain unchanged. Full Kraken archive parts were streamed, publisher-SHA256 checked, then daily members independently matched to the retained TASK001 CSV hashes. No newer quarterly archive was advertised on the refreshed official page; latest complete archive is already represented. REST data is separately attributed as provisional and must be reconciled when a new quarterly archive appears.

Absent intervals are explicitly listed, with REVIEW_REQUIRED retained. Kraken's generic no-trade explanation does not constitute independent confirmation of each historical date. No missing price is filled. Derived V4 availability follows candle close; original close-plus-publication-lag evidence is preserved. Data corruption and original-data hash tests are mandatory.
''')
    md('docs/VALIDATION_PROTOCOL_V4.md','''# Validation protocol

Frozen split table: experiments/SPLITS_V4.json. Original final decisions are touched audit material, never clean validation. DEV uses purged annual expanding folds; maximum-horizon labels must mature before the training cutoff, and scoring labels must mature inside their evaluation fold. Standardization, stopping and model selection use training data only.

Date-aggregated loss differentials account for correlated assets. DM uses horizon-minus-one HAC lag, with Holm correction. Strategy comparisons use paired calendar blocks, costed engine returns and all attempted trials in the conservative DSR calculation. Effective samples are reported as nonoverlapping windows. Exact gate outputs live in scorecard/metrics.json. The loss-registration deviation is a further promotion veto.

The frozen touched audit and fresh holdout are each consumed once; identical-manifest reruns require audit-rerun, changed manifests are rejected. A short holdout is INCONCLUSIVE_DATA, irrespective of apparent gains. Ten-year tables join separate OOS evidence segments into a labelled research index; reset accounts and purged calendar gaps prevent treating this as a continuously deployable portfolio curve.
''')
    md('docs/RISK_COST_SPEC_V4.md','''# Research risk and cost contract

Unleveraged spot BTC/USD and ETH/USD; research weights are nonnegative, capped per asset and in aggregate by configs/v4.json. These are research parameters, not approved real risk limits or a conversion of the user's verbal loss tolerance. Notional is a research USD account, not a sourced CNY conversion.

Costs have a single configuration source and are passed explicitly to every main backtest. They combine fee, modeled half-spread and slippage per side; maker scenario does not promise fills. Exact costs and sources are in config and public source receipts. Fee tiers were refreshed against Kraken's cross-platform tier support page; account tier is unconfirmed.

Targets rebalance on the fixed calendar cadence, with a no-trade band. Explicit zero targets enforce maximum holding. Positions, capacity proxy and cash are checked independently of model outputs. Freqtrade daily next-open fills do not model orderbook depth, partial fills, queue position or true intraday paths. Its emergency stop convention is not a loss guarantee. Current pair precision/min-notional metadata applied historically is a limitation.

Reconciliation uses actual engine orders and opening marks; performance comes only from engine exports. Engine wallets are sampled before daily fills. The checker tolerance is disclosed in the preregistration implementation notes and never expanded to hide a failed comparison. Constant-hold and retrospective average-exposure matched benchmarks have explicit exceptions to maximum holding.
''')

def main():
    REPORTS.mkdir(exist_ok=True);commands,test_runs=test_metrics();docs()
    score=read('experiments/v4/scorecard/metrics.json'); forecasts=read('experiments/v4/validation/metrics.json').get('forecasts',{})
    bt=read('experiments/v4/backtest_summary/metrics.json');ten=read('experiments/v4/tenyear/metrics.json')
    training=[read(str(p.relative_to(ROOT))) for p in sorted((OUT/'training').glob('dev_fold*/metrics.json'))]
    trainrows=[]
    for i,fold in enumerate(training):
        for name,t in fold.get('training',{}).items():trainrows.append([i,name,t['device'],t['parameters'],num(t['seconds']),t['epochs_completed'],num(t.get('inference_seconds'))])
    md('reports/TRAINING_V4_REPORT.md','# Actual training\n\n'+table(['Fold','Model','Device','Parameters','Seconds','Epochs','Inference seconds'],trainrows)+
       'Source: experiments/v4/training/**/metrics.json. All five seeds and every fold are retained. Final DEV-only fit: experiments/v4/frozen_model/metrics.json. Original Transformer remains RESEARCH_NEGATIVE, independently preserved under experiments/price/formal-4090-v1. No old sealed set was relabelled as fresh.\n')
    predtable=table(['Model / horizon','MAE','Zero MAE','MAE skill','HAC p','Holm p','Independent windows'],
                   [[k,num(v['mae']),num(v['zero_mae']),pct(v['mae_skill']),num(v['dm']['p_one_sided']),num(v['p_holm']),v['independent_windows']] for k,v in forecasts.items()])
    gatetable=table(['Model','G1','G2','G3','G4','BASE Sharpe delta','Bootstrap lower','DSR','STRESS delta'],
                   [[k,v['G1'],v['G2'],v['G3'],v['G4'],num(v.get('sharpe_delta')),num(v.get('bootstrap',{}).get('lower90')),num(v.get('deflated_sharpe')),num(v.get('stress_delta'))] for k,v in score.get('gates',{}).items()])
    md('reports/OOS_V4_REPORT.md','# Preregistered OOS evidence\n\n'+predtable+gatetable+
       f"Alpha conclusion: {score.get('alpha','PENDING')}. Strongest simple baseline selected on DEV only: {score.get('best_simple_baseline','PENDING')}.\n\n"+
       'Fresh terminal status: '+read('experiments/v4/FRESH_HOLDOUT/metrics.json').get('status','PENDING')+'. Actual sample/window counts and all forecasts are retained in that metrics file. No terminal pass is claimed. The fixed auxiliary-loss registration deviation independently prevents Alpha promotion.\n')
    btrows=[[k,pct(v['net_return']),num(v['sharpe']),pct(v['max_drawdown']),v['trades'],num(v['turnover_notional_over_initial']),pct(v['mean_exposure']),num(v['effective_cost_paid'])] for k,v in bt.items()]
    md('reports/BACKTEST_V4_REPORT.md','# Actual Freqtrade offline backtests\n\n'+table(['Segment / model / cost','Net return','MTM Sharpe','MTM drawdown','Trades','Turnover','Mean exposure','Effective cost paid'],btrows)+
       'Every number comes from backtest_summary/metrics.json, which references actual engine ZIPs, wallets and fill Parquet. Net return is the engine closed-account statistic; Sharpe/drawdown use engine mark-to-market wallet observations. Initial engine revision is retained but excluded because reconciliation exposed equity/holding semantics errors. Native network audit and lookahead/recursive outputs are retained. Docker network namespace was not available.\n')
    yrrows=[[v['cost'],v['model'],v['year'],v['segment'],pct(v['net_return']),num(v['sharpe']),pct(v['mean_exposure']),v['insufficient_early_training']] for v in ten.get('annual',[])]
    md('reports/TENYEAR_V4_REPORT.md','# Ten-year layered historical consistency\n\n'+table(['Cost','Model','Year','Evidence segment','Net return','Sharpe','Exposure','Insufficient early training'],yrrows)+
       'The constant-weight equal-average-exposure comparator is retrospective and is priced by the same engine; it is not a predictive strategy. Its target and realized exposure are separately reported. Recent years are isolated in tenyear/metrics.json; state splits are in scorecard/metrics.json. The five deepest buy-hold drawdown windows are selected algorithmically, not manually.\n\n'+
       json.dumps(ten.get('deepest_buy_hold_drawdowns',[]),indent=2)+'\n\nLimitations: '+ '; '.join(ten.get('limitations',[]))+'.\n')
    cmdrows=[[str(c.get('command')),c.get('exit_code'),c.get('log')] for c in commands]
    md('reports/TESTS_V4_REPORT.md','# Test and execution evidence\n\n'+table(['JUnit artifact','Tests','Failures','Errors','Skipped'],[[r['path'],r['tests'],r['failures'],r['errors'],r['skipped']] for r in test_runs])+table(['Actual argv','Actual exit code','Log'],cmdrows)+
       'Earlier failed attempts remain listed. Later passing runs do not erase those failures. Freqtrade can return a successful process code for some configuration/parser errors; semantic log checks are additionally required, and the earlier recursive-analysis attempts are NOT accepted as successful. The final recursive analysis output is the third analysis revision.\n')
    state=read('experiments/BUILD_STATE.json');p1=read('experiments/v4/p1/metrics.json');requirements=state.get('requirements',{})
    full_runs=[r for r in test_runs if r['tests']>=169]
    latest_full=max(full_runs,key=lambda r:r['modified_at']) if full_runs else None
    tests_good=latest_full is not None and not(latest_full['failures'] or latest_full['errors'] or latest_full['skipped'])
    full_receipts=[c for c in commands if '-m' in c.get('command',[]) and 'pytest' in c.get('command',[]) and not any(str(a).startswith('tests/') for a in c['command'])]
    last_test=max(full_receipts,key=lambda r:r['started_at']) if full_receipts else {}
    evidence={
      'R-GOV-01':['tests/test_v4_research.py','experiments/v4/tests-p0.xml'],
      'R-GOV-02':['experiments/v4/ledger.jsonl','experiments/BUILD_STATE.json','experiments/DECISIONS.md'],
      'R-GOV-03':['experiments/v4/report_provenance.json','scripts/v4_reports.py'],
      'R-DATA-01':['experiments/v4/archive_verification/metrics.json','experiments/v4/archive_verification/members.json','experiments/v4/data_lineage.json'],
      'R-DATA-02':['src/quant_platform/data/pit.py','experiments/v4/features.parquet'],
      'R-DATA-03':['experiments/SPLITS_V4.json','experiments/v4/registration.json'],
      'R-FAC-01':['experiments/v4/factor_registry.json'], 'R-FAC-02':['experiments/v4/factor_quality/metrics.json'],
      'R-MOD-01':['experiments/v4/backtest_summary/metrics.json','experiments/v4/dev_predictions.parquet'],
      'R-MOD-02':['experiments/v4/validation/metrics.json'], 'R-MOD-03':['experiments/v4/training','experiments/v4/frozen_model/metrics.json'],
      'R-VAL-01':['tests/test_v4_research.py','experiments/v4/validation/metrics.json'],
      'R-VAL-02':['experiments/PREREGISTRATION_V4.md','experiments/PREREGISTRATION_IMPLEMENTATION_NOTES_V4.md','experiments/v4/scorecard/metrics.json'],
      'R-VAL-03':['experiments/v4/frozen.json','experiments/FINAL_CONSUMED.json','experiments/TOUCHED_AUDIT_CONSUMED.json'],
      'R-STR-01':['src/quant_platform/strategy/decision.py','experiments/v4/signals/v2'],
      'R-BT-01':['experiments/v4/freqtrade_v2/analyses_v3/metrics.json','experiments/v4/freqtrade/network_audit.jsonl','reports/BACKTEST_V4_REPORT.md'],
      'R-BT-02':['experiments/v4/freqtrade_v2/DEV/rule/BASE/reconciliation_metrics.json','tests/test_v4_research.py'],
      'R-TEST-01':['reports/TESTS_V4_REPORT.md','experiments/v4/tests-p0.xml'],
      'R-REP-01':['reports/V4_COMPLETION_REPORT.md','reports/V4_STATUS.json'],
      'R-HYP-01':['experiments/HYPOTHESES_V4.md','experiments/v4/registration.json'],
      'R-MOD-06':['experiments/v4/freqtrade_v2/DEV/rule/BASE/metrics.json'],
      'R-VAL-05':['reports/TENYEAR_V4_REPORT.md','experiments/v4/tenyear/metrics.json']}
    for rid,item in requirements.items():
        if item['priority']=='P0':
            item.update(status='IMPLEMENTED_AND_TESTED' if tests_good else 'IMPLEMENTED_NOT_RUN',evidence=evidence.get(rid,[]),
                        command=last_test.get('command'),exit_code=last_test.get('exit_code'),
                        verified_at=now(),git_sha=git_sha(ROOT))
        elif rid in p1.get('requirements',{}):item.update(p1['requirements'][rid])
        else:item.update(status='BLOCKED',evidence=['reports/BLOCKED_OPTIONAL_V4.md'],command=None,exit_code=None,reason='Optional item awaiting prerequisite or available environment; see explicit blocker report.')
    if not any(all(forecasts.get(f'{name}/{h}',{}).get('G1_horizon_pass',False) for h in [14,28]) for name in ['ridge','elasticnet','lightgbm']):
        requirements['R-MOD-02']['research_outcome']='RESEARCH_NEGATIVE'
    if not any(v.get('G1') and v.get('G2') and v.get('G3') for v in score.get('gates',{}).values()):
        requirements['R-MOD-03']['research_outcome']='RESEARCH_NEGATIVE'
    state.update(updated_at=now(),git_sha=git_sha(ROOT),phase='COMPLETE_SAFE_SCOPE_WITH_BLOCKERS',requirements=requirements)
    write_json(ROOT/'experiments/BUILD_STATE.json',state)
    # Conservative software status: disclosed protocol deviation and unverified
    # container isolation prevent claiming complete specification conformance.
    status=dict(schema_version=1,generated_at=now(),software='PARTIAL',alpha=score.get('alpha','PENDING'),
                integration=p1.get('integration','PARTIAL_INTEGRATION'),remote_gpu='BLOCKED_REMOTE_GPU',requirements=requirements,
                software_limitations=['auxiliary tail-loss preregistration deviation','Docker network-none verification unavailable; native guarded engine used',
                                      'full historical prospective-source publication evidence incomplete',
                                      'one early Freqtrade retry overwrote a detailed engine log; failed receipt and ledger entry retained'],
                next_step='Keep the consumed holdout frozen; collect public PIT forward observations with the rule baseline. Supply a documented remote channel only if extra GPU work is needed.')
    write_json(ROOT/'reports/V4_STATUS.json',status)
    rows=[[rid,v['status'],v.get('research_outcome','—'),'; '.join(v['evidence']),str(v.get('command')),v.get('exit_code')] for rid,v in requirements.items()]
    text=f'''# V4.4 统一开发完成报告

软件：**{status['software']}**。Alpha：**{status['alpha']}**。集成：**{status['integration']}**。远程 GPU：**{status['remote_gpu']}**。

最简单的下一步：保留规则基线，继续按文件采集免费、带可用时间戳的前向数据；新模型先另开预注册，不再调整已消耗的终测区。

已完成本机可运行的研究链、真实 Lenovo GPU 训练、Freqtrade 三档成本回测与审计文件。软件保守记为 PARTIAL：尾部辅助损失未在原预注册参数表单列、Docker 网络隔离未实测、历史事件公告时点仍缺证据；这些限制没有被改写成通过。没有使用实盘账户、交易密钥、真实订单、杠杆或在线 dry-run bot。

另有审计保留缺口：早期一次 Freqtrade smoke 重试覆盖了该次详细 run.log；失败命令回执和 append-only 台账记录仍在，未冒称原始详细报错文件完整保存。后续修订目录、失败测试和独立 smoke 均保留。

旧 Transformer 失败结果原样保留为 RESEARCH_NEGATIVE，旧封存区只作已触碰审计。多因子树模型和 SIDGTN 未满足预注册预测门；成本后局部正增量不能替换 G1–G4。DEV 总结为 NO_GO_ALPHA；新终测区单独为 INCONCLUSIVE_DATA，不能宣称通过终测。下面数字均由 metrics 文件生成。

'''+gatetable+predtable+'''H1/H2 的规则、仅方向、波动目标和模型变体均有成本后对照；Kraken 历史资金费率和 DVOL 另有同区间增量检验。BitMEX、历史基差和公告时点缺口没有被其他数据冒充。H5 探索结果不参与主门。

本轮没有模型获准晋级为策略观察候选。文件型前向台账已记录失败模型的诊断预测，并与规则基准和未来新版本分轨；标签尚未成熟，不计算虚构收益。仅增加样本时间不能修复已失败的 DEV 门，模型变化必须另开注册与前向轨。

五个开源项目的实查、实际模块调用、真实命令退出码及许可见 OPEN_SOURCE_REUSE_AUDIT.md、THIRD_PARTY_NOTICES.md 和 INTEGRATION_V4_REPORT.md。Freqtrade 和 Qlib 有真实调用；FinRL-X 只参考设计；RD-Agent、TradingAgents 因环境或本地模型延期。FreqAI 仅通过可用性 smoke。远程作业协议在本机跑通，不代表当前 4090 远程通道可用。

未来 IBKR 股票需要独立的 PIT 行情、交易日历、公司行为、退市与存续偏差、合约身份、数据权限和费用模块；期权还需要历史链与报价、定价/Greeks、乘数与交割物、行权/指派、到期、抵押和独立生命周期风控。本次只加本地快照类型接口，没有开发连接或交易适配器。

完整训练、回测、样本外、十年分层和测试证据分别在相应 V4 报告；负结果、失败尝试和协议偏差均保留。默认值和可逆决策见 DECISIONS.md，单命令重放见 README_V4.md。十年图表为分证据区段拼接的研究指数；账户重置、purge/embargo 缺段和早期样本不足均有说明，不称为连续可投资曲线。

'''
    from v4_supplement_report import supplement
    extra=supplement();text+=extra['summary']+'\n逐项验收状态与证据（软件实现状态与研究负结果分开记录）：\n\n'+table(['需求 ID','状态','研究结果','证据路径','真实命令','退出码'],rows)
    text+='\n费用确认后使用 `scripts/reprice_v4.py --cost-config <cost-only-config>` 对冻结信号重计成本，无需重训。原终测预测及消费标记保持封存。\n\n成本研究指数图：reports/figures/V4_COSTED_RESEARCH_INDEX.png（仅研究拼接指数，非连续可投资净值）。\n'
    md('reports/TENYEAR_V4_REPORT.md',(ROOT/'reports/TENYEAR_V4_REPORT.md').read_text(encoding='utf-8')+'\n'+extra['tenyear'])
    md('reports/OOS_V4_REPORT.md',(ROOT/'reports/OOS_V4_REPORT.md').read_text(encoding='utf-8')+'\n'+extra['oos'])
    md('reports/V4_COMPLETION_REPORT.md',text)
    report_paths=[ROOT/'reports'/name for name in ['V4_COMPLETION_REPORT.md','TRAINING_V4_REPORT.md','BACKTEST_V4_REPORT.md','OOS_V4_REPORT.md','TENYEAR_V4_REPORT.md','TESTS_V4_REPORT.md','HYPOTHESES_V4_REPORT.md','INTEGRATION_V4_REPORT.md']]
    source_paths=[p for p in OUT.rglob('metrics.json') if p.is_file()]
    write_json(OUT/'report_provenance.json',dict(schema_version=1,generator='scripts/v4_reports.py',
               sources={p.relative_to(ROOT).as_posix():digest_file(p) for p in source_paths},reports={p.relative_to(ROOT).as_posix():digest_file(p) for p in report_paths},
               media={p.relative_to(ROOT).as_posix():digest_file(p) for p in (REPORTS/'figures').glob('V4_COSTED_RESEARCH_INDEX.*')}))
    print('REPORTS_GENERATED_FROM_METRICS')
if __name__=='__main__':main()
