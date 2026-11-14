"""Generate a new report from retained V4 metrics and the new observation track."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import Ledger, digest_file, git_sha, now, write_json
OUT=ROOT/'experiments/next_step_20261113'

def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x).replace('|','/') for x in r)+' |\n' for r in rows)+'\n'
def p(v): return 'MISSING' if v is None else f'{v*100:.2f}%'
def n(v): return 'MISSING' if v is None else f'{v:.4f}'

def main():
    source_paths=['experiments/v4/backtest_summary/metrics.json','experiments/v4/validation/metrics.json',
        'experiments/v4/scorecard/metrics.json','experiments/v4/extended/metrics.json',
        'experiments/v4/FRESH_HOLDOUT/metrics.json','experiments/v4/tenyear/metrics.json',
        'experiments/next_step_20261113/causal_dev_correction/metrics.json',
        'experiments/next_step_20261113/external_signal_truncation/metrics.json',
        'experiments/next_step_20261113/external_signal_truncation/alias_audit.json',
        'experiments/rule_forward_v1/freeze.json','experiments/rule_forward_v1/latest.json']
    latest=read('experiments/rule_forward_v1/latest.json')
    source_paths.append(latest['result']); forward=read(latest['result'])
    metrics=read(source_paths[0]); forecasts=read(source_paths[1])['forecasts']; score=read(source_paths[2])
    corrected=read('experiments/next_step_20261113/causal_dev_correction/metrics.json')
    state=read('experiments/BUILD_STATE.json')
    comparison=[]
    models=['buy_hold','sma200','rule','ridge','elasticnet','lightgbm']+[f'sidgtn_seed{s}' for s in [11,22,33,44,55]]
    for model in models:
        for cost in ['BASE','STRESS']:
            key=f'DEV/{model}/{cost}'; m=metrics.get(key,{})
            gates=score.get('gates',{}).get(model,{})
            if not gates and model in ['ridge','elasticnet','lightgbm']:
                gates={'G1':all(forecasts.get(f'{model}/{h}',{}).get('G1_horizon_pass',False) for h in [14,28])}
            comparison.append(dict(key=key,model=model,cost=cost,
                **{k:m.get(k) for k in ['net_return','sharpe','max_drawdown','turnover_notional_over_initial',
                                      'trades','independent_28d_windows','mean_exposure','effective_cost_paid']},
                **{g:gates.get(g,'NOT_EVALUATED') for g in ['G1','G2','G3','G4']},
                missing_evidence=['old execution exporter fails causal audit; only rule DEV corrected',
                                  'no Alpha promotion; auxiliary-loss registration deviation',
                                  'new track has no mature forward windows'],
                source=source_paths[0],source_key=key))
    write_json(OUT/'comparison.json',dict(rows=comparison,forecasts=forecasts,alpha=score['alpha'],
               historical_only=True,causality_caveat=True,source_hashes={x:digest_file(ROOT/x) for x in source_paths[:4]}))
    gap_map={
      'R-OBS-02':('P2','本地模型/即时新闻尚无可用证据','不影响规则权重','延期','固定本地模型 digest，并提供当时收到的免费新闻；不接付费接口'),
      'R-FAC-04':('P1','历史 announced_at vintage 缺失','历史事件因子必须保持屏蔽','延期','补齐可审计历史公告版本后另立研究'),
      'R-OSS-03':('P2','Linux/Docker/本地 chat+embed 前提未就绪','不阻塞原生离线主线','延期','普通权限可用环境及本地模型准备好后 smoke；本轮禁止安装服务'),
      'R-OSS-04':('P2','本地模型缺失','可选 TradingAgents 不可验收','延期','固定本地模型和真实本地调用证据'),
      'R-DATA-06':('P1','GDELT 已连续三次失败；无历史市场概率','该组不可作历史特征','延期','仅在正常公共接口恢复后采集真实收到的前向数据，不绕过限制'),
      'R-DATA-04':('P1','默认 BTC/ETH；上下文资产未启用','额外资产研究未运行','延期','明确额外上下文的独立研究价值后启用已实现契约')}
    gaps=[]
    for rid,item in state['requirements'].items():
        if item['status'] in ['BLOCKED','IMPLEMENTED_NOT_RUN']:
            priority,cause,impact,action,unblock=gap_map[rid]
            gaps.append([rid,priority,item['status'],'; '.join(item.get('evidence',[])),cause,impact,action,unblock])
    gaps += [
      ['R-DATA-02 / R-OBS-01','P0','FIXED_NEW_TRACK','scripts/run_rule_forward.py; tests/test_rule_forward.py','收盘时间被当接收时间','旧诊断轨不是新的干净规则证据','已修复','冻结新轨，真实响应完成后写 available_at'],
      ['R-OBS-01 / R-MON-01','P0','FIXED_NEW_TRACK','src/quant_platform/research/rule_forward.py','原结算按 available_at 精确匹配，不能处理真实延迟','成熟标签与可执行时间失真','已修复','未来可执行开盘锚定；实际收齐才结算'],
      ['R-STR-01 / R-BT-01 / R-VAL-01','P0','FIXED_RULE_DEV_ONLY','experiments/next_step_20261113/external_signal_truncation','shift(-1) 缺口；共享数组回写先前权重','旧模型执行指标有因果性缺陷；规则 DEV 修正三档无指标变化','已修复主轨；其他模型结果限探索','禁止晋级；有必要复用其他模型时先做独立因果修正，不重开 final'],
      ['R-VAL-02','P0','HISTORICAL_DEVIATION_RETAINED','experiments/PREREGISTRATION_IMPLEMENTATION_NOTES_V4.md','尾部辅助损失未在原参数表注册；初训未提交源码','无法事后恢复统计资格','阻断 Alpha 晋级','保留时间证据；新假设须先注册，本轮不训练'],
      ['R-GOV-02 / R-GOV-03','P0','HISTORICAL_AUDIT_GAP_RETAINED','reports/V4_COMPLETION_REPORT.md','一次早期 Freqtrade 详细重试日志被覆盖','该次错误详细原文不可恢复，命令回执与台账仍在','保留缺口','新轮每次单独目录，新增轨首条前提交并归档精确源码'],
      ['R-VAL-03 / R-VAL-05','P0','INCONCLUSIVE_DATA','experiments/FINAL_CONSUMED.json; experiments/rule_forward_v1','终测仅13天；新轨0独立窗口','不能支持新终测/Alpha结论','等待真实数据','原 final 不重开；约336天形成12个完整独立窗口'],
      ['R-BT-01','P1','NATIVE_GUARD_VERIFIED_CONTAINER_NOT_RUN','ops/receipt_freqtrade_entry.py','Docker network-none 未验证','只能声称原生进程网络守卫','延期','当前指令禁止安装系统服务；主线无需 Docker'],
      ['NEXT-PUBLIC','P0','BLOCKED_UNSUITABLE_HISTORY','experiments/next_step_20261113/public_history_audit.json','旧历史有许可待审二进制及个人账户路径','不能直接公开本地祖先历史','阻断该次公开推送','保留原历史；准备脱敏源包，需要单独授权新的无敏感祖先公开根提交'],
      ['NEXT-REMOTE','P2','SESSION_OBSERVED_NO_TRAINING_REQUIRED','experiments/next_step_20261113/remote_status.json','现有 RustDesk 画面可读取；没有必要的新训练','硬件/Python/CUDA 尚未探测，不能称 GPU 产物验收','无需训练','后续确有预注册必要作业时直接用现有会话核实硬件与环境']]
    write_json(OUT/'gap_audit.json',dict(columns=['需求ID','实际优先级','状态','证据','根因','影响','处理','最短解除步骤'],rows=gaps))
    test_path=ROOT/'experiments/next_step_20261113/tests-targeted-regression.xml'
    xml=ET.parse(test_path).getroot(); suites=list(xml.iter('testsuite'))
    test_counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    drift=forward['monitoring']['status']; settlements=forward['settlements']; operations=forward['operations']
    current=dict(software='PARTIAL_WITH_FIXED_FORWARD_AND_CAUSAL_RULE_AUDIT',alpha=score['alpha'],
        terminal='INCONCLUSIVE_DATA',integration='PARTIAL_INTEGRATION_NATIVE_FREQTRADE_QLIB',
        remote_gpu='RUSTDESK_SESSION_OBSERVED_HARDWARE_NOT_PROBED_NO_TRAINING',
        public_push='BLOCKED_UNSUITABLE_EXISTING_HISTORY',new_training=False,
        decision='本轮无新增训练必要',forward=forward,tests=test_counts,
        generated_at=now(),git_sha=git_sha(ROOT))
    write_json(OUT/'status.json',current)
    title='# V4 续做统一报告\n\n'
    body=(f"软件：**PARTIAL**（前向与规则因果性缺口已修复；历史协议偏差保留）。Alpha：**{score['alpha']}**；新终测：**INCONCLUSIVE_DATA**。集成：原生 Freqtrade/Qlib 已实跑，仍为 PARTIAL_INTEGRATION。远程：RustDesk 当前会话画面已读取，未探测 GPU/Python/CUDA，未训练、未验收新产物。\n\n"
       '最少下一步：普通账户每天运行 `powershell.exe -NoProfile -File .\\ops\\run_forward_daily.ps1`；调度配置已导出但未启用。公开原历史被安全门阻断，见下文。\n\n'
       f"最近真实运行完成：**{forward['completed_at']}**；首条信号实际生成：{forward['generated_at']}；数据截至收盘：{forward['latest_closed_at']}。首次采集追加1条，第二次同键采集 appended={forward['appended']}，当前共{forward['observation_count']}条。\n\n"
       f"成熟标签 **{settlements['mature_labels']}**，待结算 **{settlements['pending_labels']}**；最早14天市场到期 {settlements['earliest_maturity']}，最早完整日线结算 {settlements['earliest_closed_bar_settlement']}（还需实际收到）。有效独立28天窗口 **{forward['independent_28d_windows']}**，距离12个尚差{forward['independent_windows_remaining']}个；两币同期不双计。实际连续运维 {operations['consecutive_days']}/7 天，还需 {operations['remaining_seven_day_acceptance']} 天，约336天只是完整窗口的最低覆盖，不能预报通过。\n\n"
       f"监控：{', '.join(drift)}，PSI={json.dumps(forward['psi'])}。这是分布变化报警，不是正收益证明。金融比较状态 **{forward['finance']['status']}**：尚无可成交未来完整日线，规则/BH/等平均仓位净值指标缺失，不写零收益。初始研究现金、零成交和零已发生成本仅表示尚未开始。\n\n"
       '本轮修复并验证：真实响应接收时间；严格未来开盘；原子幂等/修订追加；部分成熟逐币逐期限结算；坏链/重复/缺口/未来值拒绝；独立组合窗口计数；每日普通用户入口。新轨首条前已有精确源码提交、代码归档、数据/配置/环境哈希。原 V4 数据、权重、报告、预注册和 final 消费标记逐项哈希核对保持不变。\n\n'
       '外部信号实测：旧导出器截断后 gap_mask 和 target_weight 不一致，原因是下一日缺口依赖及数组共享回写。新规则导出器在真实 DEV 截断点重算一致。三档 DEV 纠正结果与原规则收益/Sharpe/成交相同；每档都对账，且成交日期有真实 K 线。其他旧模型执行指标仍带审计限制，只列历史探索值，不以修复规则替它们验收。未解锁 final，未新增训练。\n\n')
    body+='修正运行 ID：CAUSAL_DEV_CORRECTION_20261113，仅 DEV。\n\n'+table(['成本','规则净收益','Sharpe','最大回撤','交易数','换手','实付模型成本'],
        [[c,p(m['net_return']),n(m['sharpe']),p(m['max_drawdown']),m['trades'],n(m['turnover_notional_over_initial']),n(m['effective_cost_paid'])] for c,m in corrected['metrics'].items()])
    body+='**本轮无新增训练必要。** 现有 Ridge/ElasticNet/LightGBM 与五个 SIDGTN 种子的两期限预测门没有通过；SIDGTN 成本后门也失败。规则 BASE/STRESS 相比低成本退化，曝光差异不能当预测增量；仅延长前向时间不能解除 NO_GO。没有足以预注册的新机制假设，因此未新增实验、未为证明连通而重复训练。\n\n'
    body+='以下数字从原 metrics 自动提取，均为 DEV 历史探索结果，独立窗口列是历史回测覆盖，不是新轨干净前向样本。收益为引擎期末账户统计；Sharpe/回撤为引擎 MTM。NOT_EVALUATED 表示没有该模型的完整门证据。\n\n'
    body+=table(['模型','成本','净收益','Sharpe','最大回撤','换手','交易数','历史28天窗口','G1','G2','G3','G4'],
        [[r['model'],r['cost'],p(r['net_return']),n(r['sharpe']),p(r['max_drawdown']),n(r['turnover_notional_over_initial']),r['trades'],r['independent_28d_windows'],r['G1'],r['G2'],r['G3'],r['G4']] for r in comparison])
    body+='缺口表（包含历史5阻塞/1未运行；实际优先级为本轮必要性，不改写原需求优先级）：\n\n'+table(['需求ID','实际优先级','状态','证据','根因','影响','处理','解除步骤'],gaps)
    body+=f"验证：本轮相关回归 {test_counts['tests']} 项，失败 {test_counts['failures']}、错误 {test_counts['errors']}、跳过 {test_counts['skipped']}；另有九组隔离历史夹具的真实 Freqtrade 引擎测试（3模型×3成本），不作金融证据。首轮夹具发现缺少前一日引擎信号行，已修复；首次失败产物在 fixture_engine_smoke，成功在 fixture_engine_smoke_v2。初次单测有1项夹具 pair 缺失错误，修复后通过，首次XML仍在。第二次外部审计比较多出元数据列的校验错误也已修复，不隐去失败。旧208项为历史报告事实，本轮没有冒称全部重跑。\n\n"
    body+='远程状态区分：原生电脑控制能力存在；当前 RustDesk 远程画面可读取；GPU/目录/Python/PyTorch/CUDA 未因无训练必要而额外操作；没有远程作业启动，也没有新产物核验。旧 BLOCKED_REMOTE_GPU 只保留在旧报告，不作为本轮网络结论。没有保存密码、连接地址或截图到工程/公开包。\n\n'
    body+='GitHub 目标：[ChrisJiang123/QUANT](https://github.com/ChrisJiang123/QUANT)。已核实公开仓库、main默认分支、当时为空且具备写权限。本地原无远端，工作分支 v4-next-step-20261113。完整旧历史扫描发现许可待审 PDF/DOCX、个人账户路径；按用户续做文件第8节停止直接公开推送，不是写权限失败。未 force、未改写历史、未自动合并。可审阅脱敏包另留本地；只有另获新公开根提交授权才发布，原本地历史保留。\n\n'
    body+='按原需求 ID 的续做状态（原轮状态保留，不能由测试数推断全部需求通过）：\n\n'+table(['需求ID','原轮状态','本轮说明/证据'],[[rid,item['status'],
        '本轮缺口表对应项；新增证据 experiments/next_step_20261113、experiments/rule_forward_v1' if any(rid in g[0] for g in gaps) else '本轮未改动；原证据 '+'; '.join(item.get('evidence',[]))] for rid,item in state['requirements'].items()])
    body+='真实日志和命令回执：experiments/next_step_20261113/commands；新轨冻结、台账和运行结果：experiments/rule_forward_v1；手动、调度、故障恢复及停用方法：docs/FORWARD_OPERATIONS.md。\n'
    report=ROOT/'reports/V4_NEXT_STEP_REPORT.md'; report.write_text(title+body,encoding='utf-8')
    write_json(OUT/'report_provenance.json',dict(generated_at=now(),generator='scripts/next_step_report.py',
               sources={x:digest_file(ROOT/x) for x in source_paths},report_sha256=digest_file(report),
               test_xml_sha256=digest_file(test_path),comparison_sha256=digest_file(OUT/'comparison.json')))
    # Append continuation instead of replacing historical 42 requirement verdicts.
    continuation=dict(status_file='experiments/next_step_20261113/status.json',
                      report='reports/V4_NEXT_STEP_REPORT.md',generated_at=now(),git_sha=git_sha(ROOT))
    state.setdefault('continuations',{})['next_step_20261113']=continuation
    write_json(ROOT/'experiments/BUILD_STATE.json',state)
    public=(title+body.split('缺口表（')[0]+ '\n公开版本仅包含源代码和脱敏研究摘要；数据、权重、私有本机证据不包含。\n'
            '每日运行和安装前提见 docs/FORWARD_OPERATIONS.md；完整历史审计在本地保留，旧 Git 历史尚未获准公开。\n')
    (ROOT/'reports/V4_NEXT_STEP_PUBLIC.md').write_text(public,encoding='utf-8')
    print('NEW_REPORT_GENERATED_WITH_SOURCE_HASHES')

if __name__=='__main__': main()
